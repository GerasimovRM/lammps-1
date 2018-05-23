/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_bond_break.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "domain.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondBreak::FixBondBreak(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix bond/break command");

  MPI_Comm_rank(world,&me);

  nevery = atoi(arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/break command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = atoi(arg[4]);
  double cutoff = atof(arg[5]);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/break command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/break command");

  cutsq = cutoff*cutoff;

  // optional keywords

  iatomtype = 0;
  iminbond = 0; // min # bonds of type btype iatom can have before it's
  inewtype = 0; // changed to inewtype
  jatomtype = 0;
  jminbond = 0;
  jnewtype = 0;

  fraction = 1.0;
  int seed = 12345;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"iparam") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix bond/break command");
      iatomtype = atoi(arg[iarg+1]);
      iminbond = atoi(arg[iarg+2]);
      inewtype = atoi(arg[iarg+3]);
      if (iatomtype < 1 || iatomtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/break command");
      if ((iminbond != -1) && (iminbond < 0))
        error->all(FLERR,"Illegal fix bond/break command");
      if (inewtype < 1 || inewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/break command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"jparam") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix bond/break command");
      jatomtype = atoi(arg[iarg+1]);
      jminbond = atoi(arg[iarg+2]);
      jnewtype = atoi(arg[iarg+3]);
      if (jatomtype < 1 || jatomtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/break command");
      if ((jminbond != -1) && (jminbond < 0))
        error->all(FLERR,"Illegal fix bond/break command");
      if (jnewtype < 1 || jnewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/break command");
      fraction = atof(arg[iarg+1]);
      seed = atoi(arg[iarg+2]);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/break command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/break command");
      iarg += 3;
    } else error->all(FLERR,"Illegal fix bond/break command");
  }

  // error check

  if (atom->molecular == 0)
    error->all(FLERR,"Cannot use fix bond/break with non-molecular systems");

  if (iatomtype == jatomtype &&
       ((iminbond != jminbond) || (inewtype != jnewtype)))
    error->all(FLERR,
               "Inconsistent iparam/jparam values in fix bond/break command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix

  comm_forward = 2;
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = NULL;
  distsq = NULL;

  remv_angles = NULL;
  nremv_local = 0;
  nremv_total = 0;

  // zero out stats

  breakcount = 0;
  breakcounttotal = 0;

  anglebreakcount = 0;
  anglebreakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondBreak::~FixBondBreak()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;

  // delete locally stored arrays

  memory->destroy(bondcount);
  memory->destroy(partner);
  memory->destroy(distsq);
  memory->destroy(remv_angles);
}

/* ---------------------------------------------------------------------- */

int FixBondBreak::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::init()
{
  // require special bonds = 0,1,1

  int flag = 0;
  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 ||
      force->special_lj[3] != 1.0) flag = 1;
  if (force->special_coul[1] != 0.0 || force->special_coul[2] != 1.0 ||
      force->special_coul[3] != 1.0) flag = 1;
  if (flag) error->all(FLERR,"Fix bond/break requires special_bonds = 0,1,1");

  // warn if angles, dihedrals, impropers are being used

  if (force->angle || force->dihedral || force->improper) {
    if (me == 0)
      error->warning(FLERR,"Broken bonds WILL alter angles "
                     "but WILL NOT alter dihedrals or impropers");
  }

  // need a half neighbor list, built whenever re-neighboring occurs

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::initial_integrate(int vflag)
{
  // rebuild bondcount array if bonds have been modified

  MPI_Barrier(world);
  if (atom->bonds_modified && (!(update->ntimestep % nevery)) &&
      atom->bonds_modified >= (update->ntimestep - nevery))
    update_bondcount();
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is the first run
  // can't do this earlier, like in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,
                       "Could not count initial bonds in fix bond/break");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 0;
  if (newton_bond) comm->reverse_comm_fix(this);

  // grow the angle removal array

  memory->grow(remv_angles,atom->nangles*3,"bond/break:remv_angles");
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,bondtype,itype,jtype,possible;
  double delx,dely,delz,rsq,min,max;
  int *slist;

  if (update->ntimestep % nevery) return;

  // need updated ghost atom positions

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it

  commflag = 0;
  comm->forward_comm_fix(this);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/break:partner");
    memory->create(distsq,nmax,"bond/break:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    distsq[i] = 0.0;
  }

  // reset the angle removal array

  for (int i = 0; i < atom->nangles*3; i++) remv_angles[i] = 0;
  nremv_local = 0;
  nremv_total = 0;

  // loop over bond list
  // setup possible partner list of bonds to break

  double **x = atom->x;
  int *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    bondtype = bondlist[n][2];
    itype = type[i1];
    jtype = type[i2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (bondtype != btype) continue;

    // check against requested atomtypes

    if (iatomtype == 0 && jatomtype == 0) {
      iatomtype = itype;
      jatomtype = jtype;

      inewtype = itype;
      jnewtype = jtype;
    }

    possible = 0;
    if ((itype == iatomtype) && (jtype == jatomtype)) {
      if (((iminbond == 0) || (bondcount[i1] >= iminbond)) &&
          ((jminbond == 0) || (bondcount[i2] >= jminbond)))
        possible = 1;
    } else if ((itype == jatomtype) && (jtype == iatomtype)) {
      if (((jminbond == 0) || (bondcount[i1] >= jminbond)) &&
          ((iminbond == 0) || (bondcount[i2] >= iminbond)))
        possible = 1;
    }
    if (!possible) continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq <= cutsq) continue;

    if (rsq > distsq[i1]) {
      partner[i1] = tag[i2];
      distsq[i1] = rsq;
    }
    if (rsq > distsq[i2]) {
      partner[i2] = tag[i1];
      distsq[i2] = rsq;
    }
  }

  // reverse comm of partner info

  commflag = 1;
  if (force->newton_bond) comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 1;
  comm->forward_comm_fix(this);

  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  int **special = atom->special;

  int **angle_type = atom->angle_type;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom2 = atom->angle_atom2;
  int **angle_atom3 = atom->angle_atom3;
  int *num_angle = atom->num_angle;

  int nbreak = 0;
  int nanglebreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // apply probability constraint
    // MIN,MAX insures values are added in same order on different procs

    if (fraction < 1.0) {
      min = MIN(probability[i],probability[j]);
      max = MAX(probability[i],probability[j]);
      if (0.5*(min+max) >= fraction) continue;
    }

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == partner[i]) {
        for (k = m; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
        }

//if (screen) fprintf(screen,"%s %i %i %i %i %i %i \n","BREAK  i j iatomtype jatomtype inewtype jnewtype",i,j,iatomtype,jatomtype,inewtype,jnewtype);
        num_bond[i]--;

        break;
      }
    }

    // remove J from special bond list for atom I
    // atom J will also do this

    slist = special[i];
    n1 = nspecial[i][0];
    n3 = nspecial[i][2];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i]) break;
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // decrement bondcount, convert atoms to new type if limit reached

    bondcount[i]--;

    if (type[i] == iatomtype) {
      if ((bondcount[i] < iminbond) || (iminbond == -1)) type[i] = inewtype;
    } else if (type[i] == jatomtype) {
      if ((bondcount[i] < jminbond) || (jminbond == -1)) type[i] = jnewtype;
    }

    // count the broken bond once

    if (tag[i] < tag[j]) nbreak++;

    // delete any angles that included the now broken bond
    // add them to the array for removal only once

    m = 0;
    while (m < atom->num_angle[i]) {
      if (  (atom->angle_atom1[i][m] == atom->tag[i] &&
             atom->angle_atom2[i][m] == atom->tag[j])
                ||
            (atom->angle_atom1[i][m] == atom->tag[j] &&
             atom->angle_atom2[i][m] == atom->tag[i])
         ) {
        if (tag[i] < tag[j]) {
          remv_angles[nremv_local++] = atom->angle_atom3[i][m];
          remv_angles[nremv_local++] = tag[i];
          remv_angles[nremv_local++] = tag[j];
        }

        n = atom->num_angle[i];
        atom->angle_type[i][m] = atom->angle_type[i][n-1];
        atom->angle_atom1[i][m] = atom->angle_atom1[i][n-1];
        atom->angle_atom2[i][m] = atom->angle_atom2[i][n-1];
        atom->angle_atom3[i][m] = atom->angle_atom3[i][n-1];
        atom->num_angle[i]--;
        if (tag[i] < tag[j]) nanglebreak++;
      } else if (  (atom->angle_atom1[i][m] == atom->tag[i] &&
                    atom->angle_atom3[i][m] == atom->tag[j])
                       ||
                   (atom->angle_atom1[i][m] == atom->tag[j] &&
                    atom->angle_atom3[i][m] == atom->tag[i])
                ) {
        if (tag[i] < tag[j]) {
          remv_angles[nremv_local++] = atom->angle_atom2[i][m];
          remv_angles[nremv_local++] = tag[i];
          remv_angles[nremv_local++] = tag[j];
        }

        n = atom->num_angle[i];
        atom->angle_type[i][m] = atom->angle_type[i][n-1];
        atom->angle_atom1[i][m] = atom->angle_atom1[i][n-1];
        atom->angle_atom2[i][m] = atom->angle_atom2[i][n-1];
        atom->angle_atom3[i][m] = atom->angle_atom3[i][n-1];
        atom->num_angle[i]--;
        if (tag[i] < tag[j]) nanglebreak++;
      } else if (  (atom->angle_atom2[i][m] == atom->tag[i] &&
                    atom->angle_atom3[i][m] == atom->tag[j])
                       ||
                   (atom->angle_atom2[i][m] == atom->tag[j] &&
                    atom->angle_atom3[i][m] == atom->tag[i])
                ) {
        if (tag[i] < tag[j]) {
          remv_angles[nremv_local++] = atom->angle_atom1[i][m];
          remv_angles[nremv_local++] = tag[i];
          remv_angles[nremv_local++] = tag[j];
        }

        n = atom->num_angle[i];
        atom->angle_type[i][m] = atom->angle_type[i][n-1];
        atom->angle_atom1[i][m] = atom->angle_atom1[i][n-1];
        atom->angle_atom2[i][m] = atom->angle_atom2[i][n-1];
        atom->angle_atom3[i][m] = atom->angle_atom3[i][n-1];
        atom->num_angle[i]--;
        if (tag[i] < tag[j]) nanglebreak++;
      } else m++;
    }
  }

  // tally stats and finish removing angles

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  if (breakcount) angle_remove();

  MPI_Allreduce(&nanglebreak,&anglebreakcount,1,MPI_INT,MPI_SUM,world);
  anglebreakcounttotal += anglebreakcount;
  atom->nangles -= anglebreakcount;

  // trigger reneighboring if any bonds were removed

  if (breakcount) next_reneighbor = update->ntimestep;

  // trigger rebuilding of the bondcount array if bonds were removed

  bigint b_mod = atom->bonds_modified;
  if (breakcount) b_mod = update->ntimestep;
  MPI_Allreduce(&b_mod,&(atom->bonds_modified),1,MPI_LMP_BIGINT,
                MPI_MAX,world);
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondBreak::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;

  if (commflag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = bondcount[j];
    }
    return 1;

  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = partner[j];
      buf[m++] = probability[j];
    }
    return 2;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 0) {
    for (i = first; i < last; i++)
      bondcount[i] = static_cast<int> (buf[m++]);

  } else {
    for (i = first; i < last; i++) {
      partner[i] = static_cast<int> (buf[m++]);
      probability[i] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondBreak::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 0) {
    for (i = first; i < last; i++)
      buf[m++] = bondcount[i];
    return 1;

  } else {
    for (i = first; i < last; i++) {
      buf[m++] = partner[i];
      buf[m++] = distsq[i];
    }
    return 2;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += static_cast<int> (buf[m++]);
    }

  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] > distsq[j]) {
        partner[j] = static_cast<int> (buf[m++]);
        distsq[j] = buf[m++];
      } else m += 2;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondBreak::grow_arrays(int nmax)
{
  memory->grow(bondcount,nmax,"bond/break:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondBreak::copy_arrays(int i, int j)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondBreak::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondBreak::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondBreak::compute_vector(int n)
{
  if (n == 1) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondBreak::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes += nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   update *bondcount when needed
------------------------------------------------------------------------- */

void FixBondBreak::update_bondcount()
{
  int i,j,m;

  for (i = 0; i < (atom->nlocal + atom->nghost); i++) bondcount[i] = 0;

  for (i = 0; i < atom->nlocal; i++)
    for (j = 0; j < atom->num_bond[i]; j++) {
      if (atom->bond_type[i][j] == btype) {
        bondcount[i]++;
        if (force->newton_bond) {
          m = atom->map(atom->bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,
                       "Could not update bondcount in fix bond/break");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 0;
  if (force->newton_bond) comm->reverse_comm_fix(this);
}

/* ----------------------------------------------------------------------
   finish removing angles from 3rd atoms (already removed from the
   two atoms in the deleted bond)
------------------------------------------------------------------------- */

void FixBondBreak::angle_remove()
{
  int a1,a2,a3,m,n,me,nlocal_other,addatom;
  me = comm->me;
  nremv_total = nremv_local;

  // communicate the remv_angles array between procs

  for (int proc = 0; proc < comm->nprocs; proc++) {
    if (proc == me) nlocal_other = nremv_local;
    MPI_Bcast(&nlocal_other,1,MPI_INT,proc,world);
    for (int i = 0; i < nlocal_other; i++) {
      if (proc == me) addatom = remv_angles[i];
      MPI_Bcast(&addatom,1,MPI_INT,proc,world);
      if (proc != me) remv_angles[nremv_total++] = addatom;
    }
  }

  // each proc remove angles for owned atoms

  for (int i = 0; i < nremv_total; i += 3) {
    if (    (atom->map(remv_angles[i]) >= 0) &&
            (atom->map(remv_angles[i]) < atom->nlocal)
       ) {
      for (int j = 0; j < atom->nlocal; j++) {
        if (atom->tag[j] != remv_angles[i]) continue;
        a1 = remv_angles[i];
        a2 = remv_angles[i+1];
        a3 = remv_angles[i+2];
        m = 0;
        while (m < atom->num_angle[j]) {
          if (  (
                   (  (atom->angle_atom1[j][m] == a2) &&
                      (atom->angle_atom2[j][m] == a3)
                   ) ||
                   (  (atom->angle_atom1[j][m] == a3) &&
                      (atom->angle_atom2[j][m] == a2)
                   )
                )

                  ||

                (
                   (  (atom->angle_atom1[j][m] == a2) &&
                      (atom->angle_atom3[j][m] == a3)
                   ) ||
                   (  (atom->angle_atom1[j][m] == a3) &&
                      (atom->angle_atom3[j][m] == a2)
                   )
                )

                  ||

                (
                   (  (atom->angle_atom2[j][m] == a2) &&
                      (atom->angle_atom3[j][m] == a3)
                   ) ||
                   (  (atom->angle_atom2[j][m] == a3) &&
                      (atom->angle_atom3[j][m] == a2)
                   )
                )
             ) {
            n = atom->num_angle[j];
            atom->angle_type[j][m] = atom->angle_type[j][n-1];
            atom->angle_atom1[j][m] = atom->angle_atom1[j][n-1];
            atom->angle_atom2[j][m] = atom->angle_atom2[j][n-1];
            atom->angle_atom3[j][m] = atom->angle_atom3[j][n-1];
            atom->num_angle[j]--;
          } else m++;
        }
      }
    }
  }
  MPI_Barrier(world);
}
