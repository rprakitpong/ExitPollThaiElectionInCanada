import matplotlib.pyplot as plt

word_counts = {'paethongtarn': 0, 'pita': 0, 'sudarat': 0, 'sereepisuth': 0, 'commoner': 0, 'jurin': 0, 'anutin': 0, 'prayun': 0, 'pravit': 0, 'korn': 0, 'others': 0}
files = ['myfile.txt', 'myfile_sun.txt']

for filename in files:
    with open(filename, 'r') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1

label_map = {
    'paethongtarn': 'Pheu Thai (Paethontarn)', 
    'pita': 'Move Forward (Pita)', 
    'sudarat': 'Thai Srang Thai (Sudarat)', 
    'sereepisuth': 'Thai Liberal (Sereepisuth)', 
    'commoner': 'Commoner', 
    'jurin': 'Democrat (Jurin)', 
    'anutin': 'Bhumjaithai (Anutin)', 
    'prayun': 'UTN (Prayut)', 
    'pravit': 'Palang Pracharat (Pravit)', 
    'korn': 'Chart Pattana Kla (Korn)', 
    'others': 'Others'}

for word, count in word_counts.items():
    print(label_map[word], count)
