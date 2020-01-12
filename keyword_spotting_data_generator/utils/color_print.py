TEXT_COLOR = {
    'BOLD' : '\033[1m',
    'UNDERLINE' : '\033[4m',
    'BLUE' : '\033[94m',
    'GREEN' : '\033[92m',
    'YELLOW' : '\033[93m',
    'RED' : '\033[91m',
    'END' : '\033[0m'
}

def print_bold(*msgs):
    print(TEXT_COLOR['BOLD'])
    print(*msgs)
    print(TEXT_COLOR['END'])

def print_undeline(*msgs):
    print(TEXT_COLOR['UNDERLINE'])
    print(*msgs)
    print(TEXT_COLOR['END'])

def print_blue(*msgs):
    print(TEXT_COLOR['BLUE'])
    print(*msgs)
    print(TEXT_COLOR['END'])

def print_green(*msgs):
    print(TEXT_COLOR['GREEN'])
    print(*msgs)
    print(TEXT_COLOR['END'])

def print_yellow(*msgs):
    print(TEXT_COLOR['YELLOW'])
    print(*msgs)
    print(TEXT_COLOR['END'])

def print_red(*msgs):
    print(TEXT_COLOR['RED'])
    print(*msgs)
    print(TEXT_COLOR['END'])
