import configparser
import os

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

UPP_OUT_PATH = config['TRACE GENERATION']['UPPAAL_OUT_PATH']
CS = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')

folder = '/'.join(UPP_OUT_PATH.split('/')[:-1])
files = os.listdir(folder)
total = 0
for file in files:
    if file.startswith(CS.upper() + '_' + CS_VERSION):
        os.remove(folder + '/' + file)
        total += 1
print('{} traces deleted.'.format(total))
