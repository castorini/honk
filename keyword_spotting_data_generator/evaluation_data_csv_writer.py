import csv

class CsvWriter():
    def __init__(self, keyword):
        self.keyword = keyword
        self.file = open(keyword+".csv", 'w')

    def write(self, rows):
        for row in rows:
            self.file.write(','.join([str(i) for i in row]) + "\n")

    def __del__(self):
        self.file.close()
