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
#include "stdlib.h"
#include "string.h"
#include "fix_evaporate.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "domain.h"
#include "region.h"
#include "comm.h"
#include "force.h"
#include "group.h"
#include "random_park.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixEvaporate::FixEvaporate(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix evaporate command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 0;

  nevery = atoi(arg[3]);
  nflux = atoi(arg[4]);
  iregion = domain->find_region(arg[5]);
  int n = strlen(arg[5]) + 1;
  idregion = new char[n];
  strcpy(idregion,arg[5]);
  int seed = atoi(arg[6]);

  if (nevery <= 0 || nflux <= 0)
    error->all(FLERR,"Illegal fix evaporate command");
  if (iregion == -1)
    error->all(FLERR,"Region ID for fix evaporate does not exist");
  if (seed <= 0) error->all(FLERR,"Illegal fix evaporate command");

  // random number generator, same for all procs

  random = new RanPark(lmp,seed);

  // optional args

  molflag = 0;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"molecule") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix evaporate command");
      if (strcmp(arg[iarg+1],"no") == 0) molflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) molflag = 1;
      else if (strcmp(arg[iarg+1],"bond") == 0) molflag = 2;
      else error->all(FLERR,"Illegal fix evaporate command");
      iarg += 2;
    } 
    else if (strcmp(arg[iarg],"atomtype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix evaporate command");
      atomtype = atoi(arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix evaporate command");
  }

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = (update->ntimestep/nevery)*nevery + nevery;
  ndeleted = 0;

  nmax = 0;
  list = NULL;
  mark = NULL;
}

/* ---------------------------------------------------------------------- */

FixEvaporate::~FixEvaporate()
{
  delete [] idregion;
  delete random;
  memory->destroy(list);
  memory->destroy(mark);
}

/* ---------------------------------------------------------------------- */

int FixEvaporate::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEvaporate::init()
{
  // set index and check validity of region

  iregion = domain->find_region(idregion);
  if (iregion == -1)
    error->all(FLERR,"Region ID for fix evaporate does not exist");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if ((mask[i] & groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot evaporate atoms in atom_modify first group");
  }

  // if molflag not set, warn if any deletable atom has a mol ID

  if (molflag == 0 && atom->molecule_flag) {
    int *molecule = atom->molecule;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (molecule[i]) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall && comm->me == 0)
      error->warning(FLERR,
                     "Fix evaporate may delete atom with non-zero molecule ID");
  }

  if (molflag && atom->molecule_flag == 0)
      error->all(FLERR,
                 "Fix evaporate molecule requires atom attribute molecule");
}

/* ----------------------------------------------------------------------
   perform particle deletion
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixEvaporate::pre_exchange()
{
  int h,i,j,k,m,iwhichglobal,iwhichlocal,iwhichtypeglobal,iwhichtypelocal;
  int ndel,ndeltopo[4];
  int *type = atom->type;
  int *listtype;
  int *marktype;
  int *bondedAtom;
  int tempcount;
  int numBonded;
  char str[128];

  if (update->ntimestep != next_reneighbor) return;

  memory->create(listtype,nmax,"evaporate:listtype");
  memory->create(marktype,nmax,"evaporate:marktype");

  // grow list and mark arrays if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(list);
    memory->destroy(mark);
    memory->destroy(listtype);
    memory->destroy(marktype);
    nmax = atom->nmax;
    memory->create(list,nmax,"evaporate:list");
    memory->create(mark,nmax,"evaporate:mark");
    memory->create(listtype,nmax,"evaporate:listtype");
    memory->create(marktype,nmax,"evaporate:marktype");
  }

  // ncount = # of deletable atoms in region that I own
  // ntypecount = # of atoms of the type to be deleted in region that I own.
  // nall = # on all procs
  // ntypeall = # of atoms of the type to be deleted on all procs
  // nbefore = # on procs before me
  // list[ncount] = list of local indices of atoms that could be deleted
  //                when type is not a consideration
  // listtype[nmax] = list of local indices of atoms that could be deleted
  //                when type *is* a consideration

  double **x = atom->x;
  int *mask = atom->mask;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;

  int ntypecount = 0;
  int ncount = 0;
  for (i = 0; i < nlocal; i++) {
    list[i] = 0;
    listtype[i] = 0;
    if (mask[i] & groupbit) {
      if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2])) {
        list[ncount++] = i;
        if (type[i] == atomtype) {
          listtype[ntypecount++] = i;
        }
      }
    }
  }
/*if (1) {
  sprintf(str,"ntypecount listtype[0,1] me %d %d %d %d\n",ntypecount,listtype[0],listtype[1],comm->me);
  error->warning(FLERR,str);
}*/

  int nall,ntypeall,nbefore,ntypebefore;
  MPI_Allreduce(&ncount,&nall,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ncount,&nbefore,1,MPI_INT,MPI_SUM,world);
  nbefore -= ncount;
  MPI_Allreduce(&ntypecount,&ntypeall,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ntypecount,&ntypebefore,1,MPI_INT,MPI_SUM,world);
  ntypebefore -= ntypecount;

  // ndel = total # of atom deletions, in or out of region
  // ndeltopo[1,2,3,4] = ditto for bonds, angles, dihedrals, impropers (i.e.
  //   total number of bond, angle, dihedral, and improper deletions.)
  // mark[] = 1 if deleted
  // marktype[] = 1 if deleted

  ndel = 0;
  for (i = 0; i < nlocal; i++) mark[i] = 0;
  for (i = 0; i < nlocal; i++) marktype[i] = 0;

  // atomic deletions
  // choose atoms randomly across all procs and mark them for deletion
  // shrink eligible list as my atoms get marked
  // keep ndel,ncount,nall,nbefore current after each atom deletion

  if (molflag == 0) {
    // Loop until we have marked at least nflux atoms for deletion or we have
    //   run out of atoms of the targeted type to delete (i.e. ntypeall==0).
    while (ntypeall && ndel < nflux) {
      // Of all atoms in the system that have the targeted type, which one will
      //   be deleted? iwhichtypeglobal in the number from 0 to ntypeall-1.
      //   Note that this is *not* an atom number in the sense of being from 0
      //   to the number of atoms in the system - 1. It is just an index number
      //   covering the list of atoms of the targeted type.
      iwhichtypeglobal = static_cast<int> (ntypeall*random->uniform());
      // If the atom to be deleted is on a process *before* my process number
      //   then decrement my record of the number of atoms of the targeted type
      //   that exist before me.
      if (iwhichtypeglobal < ntypebefore) ntypebefore--;
      // Otherwise, check if I own that atom. If so, do stuff. If not, do
      //   nothing. (Recall again that iwhichtypeglobal is not an atom number
      //   in the sense described above.)
      else if (iwhichtypeglobal < ntypebefore + ntypecount) {
        // Of the atoms that have the targeted type, which one is this one?
        //   (I.e. what is the index number from 0 to the number that I own (
        //   which is ntypesbefore+ntypecount)?)
        //   The answer is called iwhichtypelocal.
        iwhichtypelocal = iwhichtypeglobal - ntypebefore;
        // Determine the *atom number* (in the traditional sense) in the list
        //   of all atoms that I own within the selected region.
        iwhichlocal = listtype[iwhichtypelocal];
        // Mark this atom as *the* one to be deleted for this while-loop
        //   iteration. (Considering all atoms I own.)
        mark[list[iwhichlocal]] = 1;
        // Move the index number of the last atom that I own into the
        //   position of the one that is to be deleted. This overwrites the
        //   index number of the atom to be deleted (effectively deleting it).
        //   This type of action will be repeated after the outer if-else block
        //   on molflag is finished except that it will be applied to the rest
        //   of the atom information.
        list[iwhichlocal] = list[ncount-1];
        // Mark this atom the *the* one to be deleted for this while-loop
        //   iteration. (Considering only the atoms of the targeted type that
        //   I own.)
        marktype[listtype[iwhichtypelocal]] = 1;
        // Move the index number of the last atom of the targeted type into the
        //   position of the one that is to be deleted. (Similar to the above
        //   operations except applied to the list of atoms of the targeted
        //   type only.)
        listtype[iwhichtypelocal] = listtype[ntypecount-1];
        // Decrement the number of atoms that I own overall and of the targeted
        //   type.
        ncount--;
        ntypecount--;
      }
      // Increment the number of atoms to be deleted and decrement the total
      //   number of atoms in the system overall and of the targeted type. 
      ndel++;
      nall--;
      ntypeall--;
    }

  // molecule deletions
  // choose one atom in one molecule randomly across all procs
  // bcast mol ID and delete all atoms in that molecule on any proc
  // update deletion count by total # of atoms in molecule
  // shrink list of eligible candidates as any of my atoms get marked
  // keep ndel,ndeltopo,ncount,nall,nbefore current after each mol deletion

  } else if (molflag == 1) {
    int me,proc,iatom,imolecule,ndelone,ndelall;
    int *molecule = atom->molecule;

    // Initialize a count of the number of bonds, angles, dihedrals, and
    //   impropers to delete.
    ndeltopo[0] = ndeltopo[1] = ndeltopo[2] = ndeltopo[3] = 0;

    // Loop until we have marked at least nflux atoms for deletion or we have
    //   run out of atoms of the targeted type to delete (i.e. ntypeall==0).
    while (ntypeall && ndel < nflux) {

      // pick an iatom,imolecule on proc me to delete
      // Of all atoms in the system that have the targeted type, which one will
      //   be deleted? iwhichtypeglobal in the number from 0 to ntypeall-1.
      //   Note that this is *not* an atom number in the sense of being from 0
      //   to the number of atoms in the system - 1. It is just an index number
      //   covering the list of atoms of the targeted type.
      iwhichtypeglobal = static_cast<int> (ntypeall*random->uniform());
      // Determine if I own the targeted atom.
      if (iwhichtypeglobal >= ntypebefore && 
          iwhichtypeglobal < ntypebefore + ntypecount) {
        // Of the atoms that have the targeted type, which one is this one?
        //   (I.e. what is the index number from 0 to the number that I own (
        //   which is ntypesbefore+ntypecount)?)
        //   The answer is called iwhichtypelocal.
        iwhichtypelocal = iwhichtypeglobal - ntypebefore;
        // Determine the *atom number* (in the traditional sense) in the list
        //   of all atoms that I own.
        iwhichlocal = listtype[iwhichtypelocal];
//        iatom = list[iwhichlocal];
        iatom = iwhichlocal; // This deviation from the original source arises
                             //   because I'm not tracking the region concept.
        // Determine the molecule number that this atom belongs to.
        imolecule = molecule[iatom];
        // Identify who I am?
        me = comm->me;
      } else me = -1; // Label myself as a process that doesn't own the
                      //   selected atom.

      // bcast mol ID to delete all atoms from
      // if mol ID > 0, delete any atom in molecule and decrement counters
      // if mol ID == 0, delete single iatom
      // be careful to delete correct # of bond,angle,etc for newton on or off

      // Identify the value of "me" that is maximal and share with all
      //   processes. (I.e. let every process know which process holds the
      //   targeted atom.) Then broadcast the molecule number from the process
      //   that holds the atom to all other processes.
      MPI_Allreduce(&me,&proc,1,MPI_INT,MPI_MAX,world);
      MPI_Bcast(&imolecule,1,MPI_INT,proc,world);
      ndelone = 0;
      // Consider all atoms that I own.
      for (i = 0; i < nlocal; i++) {
        // As mentioned above: If the imolecule ID number is == 0, then we
        //   just delete that atom or do nothing. If the imolecule ID number is
        //   > 0 then we delete any atom of the same molecule.
        if (imolecule && molecule[i] == imolecule) {
          // Mark this atom as one to be deleted and increment the count of the
          //   number of atoms to delete.
          mark[i] = 1;
          ndelone++;

          // If the atom_style allows for the definition of bonds...
          if (atom->avec->bonds_allow) {
            if (force->newton_bond) ndeltopo[0] += atom->num_bond[i];
            else {
              // At this point we have found an atom (i) that is in the same
              //   molecule as the atom targeted for deletion and so this atom
              //   (i) should also be deleted. So, we need to delete all of the
              //   bonds associated with atom (i). However, the trick that is
              //   applied is that we don't explicitly delete any bonds, rather
              //   we just copy over the atom in index (i) with the atom from
              //   the end of the local array of atoms. Therefore, we only need
              //   to keep track of the total number of bonds etc. that are
              //   "deleted". Because newton bonds are "off" the number of
              //   bonds that are removed needs to be obtained carefully. That
              //   is because every bond is actually recorded twice, once for
              //   each atom in the bond. So, on this process for this (i) loop
              //   iteration we only count those bonded atoms that have a
              //   global index number that is higher than the current one for
              //   (i) which is tag[i]. The expectation is that the atoms on
              //   the other side of the bond will remove the other record of
              //   the bond when they get to this point. HOWVER, the assumption
              //   is that the bonded atoms are both part of the same molecule.
              //   This may not need to be the case and so I would consider
              //   this aspect of the program to be a bit broken. USE WITH
              //   CAUTION! Once the number of bonds to be deleted (ndeltopo[0])
              //   is obtained it is really only used to adjust the record of
              //   the total number of bonds that exist in the system because
              //   all the bonds themselves are (as mentioned previously) not
              //   explicitly deleted. Rather, the atom to be deleted is marked
              //   and then copied over later (destroying the record of the
              //   bonds in the process).
              for (j = 0; j < atom->num_bond[i]; j++) {
                if (tag[i] < atom->bond_atom[i][j]) ndeltopo[0]++;
              }
            }
          }
          // Basically the same as for the bonds.
          if (atom->avec->angles_allow) {
            if (force->newton_bond) ndeltopo[1] += atom->num_angle[i];
            else {
              for (j = 0; j < atom->num_angle[i]; j++) {
                m = atom->map(atom->angle_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[1]++;
              }
            }
          }
          // Basically the same as for the bonds.
          if (atom->avec->dihedrals_allow) {
            if (force->newton_bond) ndeltopo[2] += atom->num_dihedral[i];
            else {
              for (j = 0; j < atom->num_dihedral[i]; j++) {
                m = atom->map(atom->dihedral_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[2]++;
              }
            }
          }
          // Basically the same as for the bonds.
          if (atom->avec->impropers_allow) {
            if (force->newton_bond) ndeltopo[3] += atom->num_improper[i];
            else {
              for (j = 0; j < atom->num_improper[i]; j++) {
                m = atom->map(atom->improper_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[3]++;
              }
            }
          }

        } else if (me == proc && i == iatom) {
          mark[i] = 1;
          ndelone++;
        }
      }

      // remove any atoms marked for deletion from my eligible list

      i = 0;
      tempcount = 0;
      while (i < ncount) {
        if (mark[list[i]]) {
          list[i] = list[ncount-1];
          ncount--;
          if (type[i] == atomtype) {
            listtype[tempcount++] = listtype[ntypecount-1];
            ntypecount--;
          }
        } else {
          if (type[i] == atomtype) tempcount++;
          i++;
        }
      }

      // update ndel,ncount,nall,nbefore
      // ndelall is total atoms deleted on this iteration
      // ncount is already correct, so resum to get nall and nbefore

      MPI_Allreduce(&ndelone,&ndelall,1,MPI_INT,MPI_SUM,world);
      ndel += ndelall;
      MPI_Allreduce(&ncount,&nall,1,MPI_INT,MPI_SUM,world);
      MPI_Allreduce(&ntypecount,&ntypeall,1,MPI_INT,MPI_SUM,world);
      MPI_Scan(&ncount,&nbefore,1,MPI_INT,MPI_SUM,world);
      MPI_Scan(&ntypecount,&ntypebefore,1,MPI_INT,MPI_SUM,world);
      nbefore -= ncount;
      ntypebefore -= ntypecount;
    }
  } else if (molflag == 2) {
    int me,proc,iatom,ndelone,ndelall;
/*if ((comm->me == 0) && (ntypeall > 0)) {
  sprintf(str,"Attempting a bonded atom removal\n");
  error->warning(FLERR,str);
}*/

    ndeltopo[0] = ndeltopo[1] = ndeltopo[2] = ndeltopo[3] = 0;

    while (ntypeall && ndel < nflux) {
      // pick an atom to delete.

      // This will proceed pretty much just like the molflag==1 case above
      //   except that we will *not* delete other atoms on the basis of
      //   belonging to the same molecule. Instead we will delete atoms on the
      //   basis of being bonded to the selected atom. At the current time,
      //   the deletion will *not* proceed recursively. That is, only one level
      //   of bonding is followed so that atoms bonded to the atom that is
      //   bonded to the atom that is to be deleted will not be deleted. (Read
      //   that a couple of times.)
      iwhichtypeglobal = static_cast<int> (ntypeall*random->uniform());
      if (iwhichtypeglobal >= ntypebefore && 
          iwhichtypeglobal < ntypebefore + ntypecount) {
        iwhichtypelocal = iwhichtypeglobal - ntypebefore;
        iwhichlocal = listtype[iwhichtypelocal];
        iwhichglobal = tag[iwhichlocal];
//        iatom = list[iwhichlocal];
        iatom = iwhichlocal;
        numBonded = atom->num_bond[iatom];
        me = comm->me;
      } else me = -1;
/*if (comm->me == 0) {
  sprintf(str,"ntypeall ndel nflux iatom iwhichglobal numBonded %d %d %d %d %d %d\n",ntypeall,ndel,nflux,iatom,iwhichglobal,numBonded);
  error->warning(FLERR,str);
}*/
      // be careful to delete correct # of bond,angle,etc for newton on or off

      // Let all processes know which process owns the targeted atom.
      MPI_Allreduce(&me,&proc,1,MPI_INT,MPI_MAX,world);

      // bcast the global atom ID of the targeted atom.
      MPI_Bcast(&iwhichglobal,1,MPI_INT,proc,world);

      // bcast the number of atoms bonded to the targeted atom
      MPI_Bcast(&numBonded,1,MPI_INT,proc,world);

      // Make a list of the global unique ID numbers of the atoms bonded to
      //   the target atom plus the bonded atom itself.
      memory->create(bondedAtom,numBonded+1,"evaporate:bondedatom");
      // bcast the global atom IDs of all atoms bonded to the targeted atom.
      for (i = 0; i < numBonded; i++) {
        if (me > -1) {bondedAtom[i] = atom->bond_atom[iatom][i];}
        else {bondedAtom[i] = 0;}
        MPI_Bcast(&bondedAtom[i],1,MPI_INT,proc,world);
//sprintf(str,"i numBonded bondedAtom[i] %d %d %d \n",i,numBonded,bondedAtom[i]);
//error->warning(FLERR,str);
      }
      // Add the targeted atom to the list of bonded atoms.
      if (me > -1) {bondedAtom[numBonded] = atom->tag[iatom];}
      MPI_Bcast(&bondedAtom[numBonded],1,MPI_INT,proc,world);
      numBonded++; // We have added the targeted atom itself to the bonded list.
//sprintf(str,"numBonded bondedAtom[numBonded] %d %d \n",numBonded,bondedAtom[numBonded]);
//error->warning(FLERR,str);
      ndelone = 0;

      // if this process owns any atom that is bonded to the target atom then
      // mark it for deletion.

      // Every process knows the global ID number of the atom to be deleted
      //   and how many bonds it has and the global ID numbers of those bonded
      //   atoms.
      // Now, every process will determine if it owns one of the bonded atoms
      //   and if so, it will mark it for deletion.
      for (h = 0; h < numBonded; h++) {
        // Given the global unique atom ID of the bonded atom, determine the
        //   local ID number and if I own it, do stuff.
        k = atom->map(bondedAtom[h]);
//sprintf(str,"h k bondedAtom[h] nlocal %d %d %d %d \n",h,k,bondedAtom[h],nlocal);
//error->warning(FLERR,str);
        if (k >= 0 && k < nlocal) {
          // Mark the atom for deletion (by copy overwrite).
          mark[k] = 1;
          ndelone++;

          i = atom->map(bondedAtom[h]);

          // Count the bonds, angle, dihedrals, and impropers that will be
          //   removed by the copy overwrite.
          if (atom->avec->bonds_allow) {
            if (force->newton_bond) ndeltopo[0] += atom->num_bond[i];
            else {
              for (j = 0; j < atom->num_bond[i]; j++) {
                if (tag[i] < atom->bond_atom[i][j]) ndeltopo[0]++;
              }
            }
          }
          if (atom->avec->angles_allow) {
            if (force->newton_bond) ndeltopo[1] += atom->num_angle[i];
            else {
              for (j = 0; j < atom->num_angle[i]; j++) {
                m = atom->map(atom->angle_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[1]++;
              }
            }
          }
          if (atom->avec->dihedrals_allow) {
            if (force->newton_bond) ndeltopo[2] += atom->num_dihedral[i];
            else {
              for (j = 0; j < atom->num_dihedral[i]; j++) {
                m = atom->map(atom->dihedral_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[2]++;
              }
            }
          }
          if (atom->avec->impropers_allow) {
            if (force->newton_bond) ndeltopo[3] += atom->num_improper[i];
            else {
              for (j = 0; j < atom->num_improper[i]; j++) {
                m = atom->map(atom->improper_atom2[i][j]);
                if (m >= 0 && m < nlocal) ndeltopo[3]++;
              }
            }
          }
        }
      }

      // remove any atoms marked for deletion from my eligible list

      i = 0;
      tempcount = 0;
      while (i < ncount) {
        if (mark[list[i]]) {
          list[i] = list[ncount-1];
          ncount--;
          if (type[i] == atomtype) {
            listtype[tempcount++] = listtype[ntypecount-1];
            ntypecount--;
          }
        } else {
          if (type[i] == atomtype) tempcount++;
          i++;
        }
      }

      // update ndel,ncount,nall,nbefore
      // ndelall is total atoms deleted on this iteration
      // ncount is already correct, so resum to get nall and nbefore

      MPI_Allreduce(&ndelone,&ndelall,1,MPI_INT,MPI_SUM,world);
      ndel += ndelall;
      MPI_Allreduce(&ncount,&nall,1,MPI_INT,MPI_SUM,world);
      MPI_Allreduce(&ntypecount,&ntypeall,1,MPI_INT,MPI_SUM,world);
      MPI_Scan(&ncount,&nbefore,1,MPI_INT,MPI_SUM,world);
      MPI_Scan(&ntypecount,&ntypebefore,1,MPI_INT,MPI_SUM,world);
      nbefore -= ncount;
      ntypebefore -= ntypecount;

      memory->destroy(bondedAtom);
    }
  }

  // At this point we will delete my marked atoms
  // loop in reverse order to avoid copying marked atoms

  AtomVec *avec = atom->avec;

  for (i = nlocal-1; i >= 0; i--) {
    if (mark[i]) {
      // Note that this copy call is virtual and thus specific to the
      //   atom_style defined in the LAMMPS input file. As an example, if the
      //   atom_style is "full", then all bond, angle, dihedral, and improper
      //   information is also copied.
      avec->copy(atom->nlocal-1,i,1);
      atom->nlocal--;
    }
  }

  // reset global natoms and bonds, angles, etc
  // if global map exists, reset it now instead of waiting for comm
  // since deleting atoms messes up ghosts

  atom->natoms -= ndel;
  if (molflag) {
    int all[4];
    MPI_Allreduce(ndeltopo,all,4,MPI_INT,MPI_SUM,world);
    atom->nbonds -= all[0];
    atom->nangles -= all[1];
    atom->ndihedrals -= all[2];
    atom->nimpropers -= all[3];
  }

  if (ndel && atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  // statistics

  ndeleted += ndel;
  next_reneighbor = update->ntimestep + nevery;

  memory->destroy(listtype);
}

/* ----------------------------------------------------------------------
   return number of deleted particles
------------------------------------------------------------------------- */

double FixEvaporate::compute_scalar()
{
  return 1.0*ndeleted;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixEvaporate::memory_usage()
{
  double bytes = 2*nmax * sizeof(int);
  return bytes;
}
