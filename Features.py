import pandas as pd

'''
This class is responsible for loading in the data and preprocessing features for use with various models.
'''
class Features:

    '''
    Initialize the list of feature names, binary mappings, and load the datasets.
    '''
    def __init__(self):
        # Define the list of features based on the dataset attributes descriptions found below.
        self.features = [
            'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 
            'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3'
        ]

        # Mapping all binary features manually.
        self.binary_mappings = {
            'school':   {'GP': 1, 'MS': 0},
            'sex':      {'F': 1, 'M': 0},
            'address':  {'U': 1, 'R': 0},
            'famsize':  {'LE3': 1, 'GT3': 0},
            'Pstatus':  {'T': 1, 'A': 0},
            'nursery':  {'yes': 1, 'no': 0},
            'higher':   {'yes': 1, 'no': 0},
            'internet': {'yes': 1, 'no': 0},
            'romantic': {'yes': 1, 'no': 0}
        }

        # Load in training and testing datasets.
        self.df_train = pd.read_csv("data//assign3_students_train.txt", sep='\t', names=self.features)
        self.df_test = pd.read_csv("data//assign3_students_test.txt", sep='\t', names=self.features)

    '''
    Preprocess the data by mapping and parsing the complex edusupport feature.

    Returns:
        df_train (dataframe): The preprocessed training DataFrame.
        df_test (dataframe): The preprocessed testing DataFrame.
    '''
    def preprocess(self, predict_16=False):   
        for col, mapping in self.binary_mappings.items():
            if col in self.df_train.columns:
                self.df_train[col] = self.df_train[col].map(mapping)
            if col in self.df_test.columns:
                self.df_test[col] = self.df_test[col].map(mapping)
        
        y_labels_train = self.df_train['edusupport'].apply(lambda x: pd.Series(self.parse_edusupport(x)))
        y_labels_test = self.df_test['edusupport'].apply(lambda x: pd.Series(self.parse_edusupport(x)))

        if (predict_16):
            self.df_train = self.df_train.drop('edusupport', axis=1)
            self.df_test = self.df_test.drop('edusupport', axis=1)
        else:
            self.df_train = pd.concat([self.df_train.drop('edusupport', axis=1), y_labels_train], axis=1)
            self.df_test = pd.concat([self.df_test.drop('edusupport', axis=1), y_labels_test], axis=1)

        return self.df_train, self.df_test, y_labels_train, y_labels_test

    '''
    Parse the complex edusupport feature.
    '''
    def parse_edusupport(self, val):
        # Handle cases where val might be NaN or not a string.
        supports = str(val).lower()
        return {
            'edu_school': 1 if 'school' in supports else 0,
            'edu_family': 1 if 'family' in supports else 0,
            'edu_paid': 1 if 'paid' in supports else 0,
            'edu_no': 1 if 'no' in supports else 0
        }
    
#**************************************** Features ****************************************#
# 1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 2. sex - student's sex (binary: 'F' - female or 'M' - male)
# 3. age - student's age (numeric: from 15 to 22)
# 4. address - student's home address type (binary: 'U' - urban or 'R' - rural)
# 5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# 6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# 7. Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th
# to 9th grade, 3 - secondary education or 4 - higher education)
# 8. Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to
# 9th grade, 3 - secondary education or 4 - higher education)
# 9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g.
# administrative or police), 'at_home' or 'other')
# 10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g.
# administrative or police), 'at_home' or 'other')
# 11. reason - reason to choose this school (nominal: close to 'home', school 'reputation',
# 'course' preference or 'other')
# 12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# 13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30
# min. to 1 hour, or 4 - >1 hour)
# 14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours,
# or 4 - >10 hours)
# 15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 16. edusupport - student receive extra educational support (nominal: 'school' (extra
# educational support from school), 'family' (from family), 'paid' (extra paid Portuguese
# classes) or 'no' (no extra educational support))
# 17. nursery - attended nursery school (binary: yes or no)
# 18. higher - wants to take higher education (binary: yes or no)
# 19. internet - Internet access at home (binary: yes or no)
# 20. romantic - with a romantic relationship (binary: yes or no)
# 21. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 22. freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 23. goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 24. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 25. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 26. health - current health status (numeric: from 1 - very bad to 5 - very good)
# 27. absences - number of school absences (numeric: from 0 to 93)
# 28. G3 - final grade (numeric: from 0 to 20, output target)