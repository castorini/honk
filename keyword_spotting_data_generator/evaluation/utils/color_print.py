TEXT_COLOUR = {
    'HEADER' : '\033[95m',
    'OKBLUE' : '\033[94m',
    'OKGREEN' : '\033[92m',
    'WARNING' : '\033[93m',
    'FAIL' : '\033[91m',
    'ENDC' : '\033[0m',
    'BOLD' : '\033[1m',
    'UNDERLINE' : '\033[4m'
}

def print_bold(*msgs):
    print(TEXT_COLOUR['BOLD'])
    print(*msgs)
    print(TEXT_COLOUR['ENDC'])

def print_progress(*msgs):
    print(TEXT_COLOUR['OKGREEN'])
    print("[ PROGRESS ] :: ", *msgs)
    print(TEXT_COLOUR['ENDC'])

def print_instruction(*msgs):
    print(TEXT_COLOUR['BOLD'])
    print(*msgs)
    print(TEXT_COLOUR['ENDC'])

def print_warning(*msgs):
    print(TEXT_COLOUR['WARNING'])
    print("[ WARNING ] :: ", *msgs)
    print(TEXT_COLOUR['ENDC'])

def print_error(*msgs):
    print(TEXT_COLOUR['FAIL'])
    print("[ ERROR ] :: ", *msgs)
    print(TEXT_COLOUR['ENDC'])
