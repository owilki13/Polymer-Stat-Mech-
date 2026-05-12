#!/bin/bash

if [ -d temp ]; then
    rm -r temp
fi

packmol < pack.inp

python3 /orcd/pool/008/zpsmith_shared/software/LUNAR/atom_typing.py -topo pack.pdb -ff PCFF-IFF

python3 /orcd/pool/008/zpsmith_shared/software/LUNAR/all2lmp.py -topo pack_typed.data -nta pack_typed.nta -class 2 -frc /orcd/pool/008/zpsmith_shared/software/LUNAR/frc_files/pcff_interface_v1_6mBN.frc

cp pack_typed_IFF.data pack.data
mkdir temp
mv pack_typed* ./temp
