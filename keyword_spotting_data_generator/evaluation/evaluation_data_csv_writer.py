class CsvWriter():
    def __init__(self, keyword, file_name=None):
        self.keyword = keyword
        if file_name:
            self.file = open(file_name, 'a+')
        else:
            self.file = open(keyword+".csv", 'w')

    def write(self, rows):
        for row in rows:
            self.file.write(','.join([str(i) for i in row]) + "\n")

    def __del__(self):
        self.file.close()
