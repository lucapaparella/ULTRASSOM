import h5py
# Caminho do arquivo
path = r"C:\Users\luca37994\Downloads\archive_to_download\archive_to_download\database\simulation\resolution_distorsion\resolution_distorsion_simu_dataset_rf.hdf5"
# Abrindo o arquivo
arquivo = h5py.File(path, "r")

print("EXPLORANDO O ARQUIVO HDF5".center(64))
print("*"*64)

def nome_e_tipo(nome, obj):
  if isinstance(obj, h5py.Group):
      print(f"GRUPO => {nome}")
  elif isinstance(obj, h5py.Dataset):
      print(f"DATASET => {nome.center(25)} | {str(obj.shape).center(15)} | {obj.dtype}")

arquivo.visititems(nome_e_tipo)

print("*"*64)