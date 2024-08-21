import os
import re
import mne
import numpy as np
def getDerivatives(pathtoDerovatives):
    
    pattern = re.compile(r'.*eeg-epo\.fif$')
    basePath = "derivatives"
    subdirs = os.listdir(basePath)
    files = []
    for s in subdirs:
        ses_dirs = os.listdir(os.path.join(basePath, s))
        
        for f in ses_dirs:
            fif_files = os.listdir(os.path.join(basePath, s, f))
            
            matching_files = [file for file in fif_files if pattern.match(file)]
            
            for file_name in matching_files:
                full_path = os.path.join(basePath, s, f, file_name)
                files.append(full_path)
    return files
                
    
def readEpoFif(filePath):
    epochs = mne.read_epochs(filePath, preload=True)
    print(epochs)
    montage = mne.channels.make_standard_montage('biosemi128') 
    epochs.set_montage(montage)
    data = []
    # Visualizar la disposición de los electrodos
    #epochs.plot_sensors(kind='topomap', show_names=True)
    epochsData = epochs.get_data()
    events = epochs.events
    eventsId = epochs.event_id
    #print(f"Datos de epochs {epochsData.shape} \n Eventos {events} \n Id de eventos {eventsId}")    
    arriba_data = epochs['Arriba'].get_data()
    abajo_data = epochs['Abajo'].get_data()
    derecha_data = epochs['Derecha'].get_data()
    izquierda_data = epochs['Izquierda'].get_data()
    print(f"INFORME DE LAS CLASES______________")
    print(f"Arriba {arriba_data.shape}")
    print(f"Abajo {abajo_data.shape}")
    print(f"Derecha {derecha_data.shape}")
    print(f"Izquierda {izquierda_data.shape}")
    data.append(arriba_data)
    data.append(abajo_data)
    data.append(derecha_data)
    data.append(izquierda_data)
    return data


def getClasses(files,saveDir):
    classes = []
    arribaClass = np.empty((0,128,1153))
    abajoClass = np.empty((0,128,1153))
    derechaClass = np.empty((0,128,1153))
    izquierdaClass = np.empty((0,128,1153))
    for f in files:
        data = readEpoFif(f)
        arribaClass = np.concatenate((arribaClass,data[0]),axis = 0)
        abajoClass = np.concatenate((abajoClass,data[1]),axis = 0)
        derechaClass = np.concatenate((derechaClass,data[2]),axis = 0)
        izquierdaClass = np.concatenate((izquierdaClass,data[3]),axis = 0)
    classes.append(arribaClass)
    classes.append(abajoClass)
    classes.append(derechaClass)
    classes.append(izquierdaClass)
    os.makedirs(saveDir, exist_ok=True)
    arriba_path = os.path.join(saveDir, "arribaData.npy")
    np.save(arriba_path, arribaClass)
    abajo_path = os.path.join(saveDir, "abajoData.npy")
    np.save(abajo_path, abajoClass)
    derecha_path = os.path.join(saveDir, "derechaData.npy")
    np.save(derecha_path, derechaClass)
    izquierda_path = os.path.join(saveDir, "izquierdaData.npy")
    np.save(izquierda_path, izquierdaClass)


def loadClasses(save_dir):
    classes = {}
    for condition in ['Arriba', 'Abajo', 'Derecha', 'Izquierda']:
        file_path = os.path.join(save_dir, f"{condition.lower()}Data.npy")
        if os.path.exists(file_path):
            classes[condition] = np.load(file_path)
            print(f"{condition} data loaded with shape: {classes[condition].shape}")
        else:
            print(f"No se encontró el archivo para la clase '{condition}'")
            classes[condition] = np.empty((0, 0, 0))
    
    return classes


