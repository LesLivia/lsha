import os
import configparser

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

UPP_OUT_PATH = config['TRACE GENERATION']['UPPAAL_OUT_PATH']
CS = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = config['SUL CONFIGURATION']['CS_VERSION'][0]

folder = '/'.join(UPP_OUT_PATH.split('/')[:-1])
files = os.listdir(folder)
for file in files:
    if file.startswith(CS.upper() + '_' + CS_VERSION):
        os.remove(folder + '/' + file)
