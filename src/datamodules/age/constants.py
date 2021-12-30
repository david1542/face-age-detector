# Age labels
AGE_LABELS = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)',
              '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


LABEL_2_INDEX = {label: i for i, label in enumerate(AGE_LABELS)}
INDEX_2_LABEL = {i: label for i, label in enumerate(AGE_LABELS)}
