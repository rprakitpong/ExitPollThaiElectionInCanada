import os
import shutil
import keyboard
import datetime
import time

key_map = {
    # bottom row
    'z': 'paethongtarn', # พรรคเพื่อไทย
    'c': 'pita', # พรรคก้าวไกล
    'b': 'sudarat', # พรรคไทยสร้างไทย
    'm': 'sereepisuth', # พรรคเสรีรวมไทย
    '.': 'commoner', # พรรคสามัญชน
    # top row
    'q': 'jurin', # พรรคประชาธิปัตย์
    'e': 'anutin', # พรรคภูมิใจไทย
    't': 'prayut', # พรรครวมไทยสร้างชาติ
    'u': 'pravit', # พรรคพลังประชารัฐ
    'o': 'korn', # พรรคชาติพัฒนากล้า
    '[': 'others', # others
    # exit
    '\\': 'exit', # exit script
}

# console_map = {
#     # bottom row
#     'z': 'พรรคเพื่อไทย',  # paethongtarn
#     'c': 'พรรคก้าวไกล',  # pita
#     'b': 'พรรคไทยสร้างไทย',  # sudarat
#     'm': 'พรรคเสรีรวมไทย',  # sereepisuth
#     '.': 'พรรคสามัญชน',  # commoner
#     # top row
#     'q': 'พรรคประชาธิปัตย์',  # jurin
#     'e': 'พรรคภูมิใจไทย',  # anutin
#     't': 'พรรครวมไทยสร้างชาติ',  # prayut
#     'u': 'พรรคพลังประชารัฐ',  # pravit
#     'o': 'พรรคชาติพัฒนากล้า',  # korn
#              '[': 'อื่นๆ',  # others
# }

filename = 'myfile_sun.txt'

def backup_file():
    now = datetime.datetime.now()
    backup_name = f'myfile_backup_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    shutil.copy(filename, backup_name)
    print(f'Created backup file: {backup_name}')

last_backup_time = datetime.datetime.now()
i = 0
while True:
    key = keyboard.read_event()
    key_name = key.name.toLowerCase()
    if key.event_type == 'down' and key_name in key_map:
        if key_name == '\\':
            exit()
        now = datetime.datetime.now()
        value = key_map[key_name]
        line = f'[{now}] {value}'
        print(str(now) + " : " + str(i))
        with open(filename, 'a') as f:
            f.write(f'{line}\n')
        if now - last_backup_time >= datetime.timedelta(minutes=30):
            backup_file()
            last_backup_time = now
        i = i + 1    
        
