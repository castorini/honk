import os

class CsvWriter():
    def __init__(self, keyword, folder_path, file_name=None):
        self.keyword = keyword
        if file_name:
            file_path = os.path.join(folder_path, file_name)
            self.file = open(file_path, 'a+')
        else:
            file_path = os.path.join(folder_path, keyword+".csv")
            print(file_path)
            self.file = open(file_path, 'w')

    def write(self, rows):
        for row in rows:
            self.file.write(','.join([str(i) for i in row]) + "\n")

    def __del__(self):
        self.file.close()
