#Imports libs
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

netdata = pd.read_csv('normal.csv', delimiter=',', decimal=',', encoding='cp1252')

y = netdata.iloc[:, :-1]
x = netdata["35"]



enc = preprocessing.LabelEncoder()
x_enc = enc.fit_transform(x)



training_inputs = y[:950]
training_outputs = x_enc[:950]
testing_inputs = y[950:]
testing_outputs = x_enc[950:]

# print(training_outputs)

from sklearn import tree

# Create the tree.DecisionTreeClassifier() scikit-learn classifier
classifier = tree.DecisionTreeClassifier()

# Train the model
classifier.fit(training_inputs, training_outputs)





from pysnmp import hlapi
import sched, time
import csv

def construct_object_types(list_of_oids):
    object_types = []
    for oid in list_of_oids:
        object_types.append(hlapi.ObjectType(hlapi.ObjectIdentity(oid)))
    return object_types


def construct_value_pairs(list_of_pairs):
    pairs = []
    for key, value in list_of_pairs.items():
        pairs.append(hlapi.ObjectType(hlapi.ObjectIdentity(key), value))
    return pairs


def get(target, oids, credentials, port=161, engine=hlapi.SnmpEngine(), context=hlapi.ContextData()):
    handler = hlapi.getCmd(
        engine,
        credentials,
        hlapi.UdpTransportTarget((target, port)),
        context,
        *construct_object_types(oids)
    )
    return fetch(handler, 1)[0]


def set(target, value_pairs, credentials, port=161, engine=hlapi.SnmpEngine(), context=hlapi.ContextData()):
    handler = hlapi.setCmd(
        engine,
        credentials,
        hlapi.UdpTransportTarget((target, port)),
        context,
        *construct_value_pairs(value_pairs)
    )
    return fetch(handler, 1)[0]


def get_bulk(target, oids, credentials, count, start_from=0, port=161,
             engine=hlapi.SnmpEngine(), context=hlapi.ContextData()):
    handler = hlapi.bulkCmd(
        engine,
        credentials,
        hlapi.UdpTransportTarget((target, port)),
        context,
        start_from, count,
        *construct_object_types(oids)
    )
    return fetch(handler, count)


def get_bulk_auto(target, oids, credentials, count_oid, start_from=0, port=161,
                  engine=hlapi.SnmpEngine(), context=hlapi.ContextData()):
    count = get(target, [count_oid], credentials, port, engine, context)[count_oid]
    return get_bulk(target, oids, credentials, count, start_from, port, engine, context)


def cast(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            try:
                return str(value)
            except (ValueError, TypeError):
                pass
    return value


def fetch(handler, count):
    result = []
    for i in range(count):
        try:
            error_indication, error_status, error_index, var_binds = next(handler)
            if not error_indication and not error_status:
                items = {}
                for var_bind in var_binds:
                    items[str(var_bind[0])] = cast(var_bind[1])
                result.append(items)
            else:
                raise RuntimeError("Got SNMP error: {0}".format(error_indication))
        except StopIteration:
            break
    return result




row = ["1.3.6.1.2.1.2.2.1.10.7", "1.3.6.1.2.1.2.2.1.16.7", "1.3.6.1.2.1.2.2.1.19.7",
"1.3.6.1.2.1.2.2.1.11.7", "1.3.6.1.2.1.2.2.1.12.7", "1.3.6.1.2.1.2.2.1.13.7", "1.3.6.1.2.1.2.2.1.17.7", 
"1.3.6.1.2.1.2.2.1.18.7", "1.3.6.1.2.1.4.3.0", "1.3.6.1.2.1.4.9.0 ",
"1.3.6.1.2.1.4.10.0" ,
"1.3.6.1.2.1.4.11.0" ,
"1.3.6.1.2.1.4.8.0",
"1.3.6.1.2.1.4.6.0",
"1.3.6.1.2.1.4.12.0",
"1.3.6.1.2.1.4.5.0","1.3.6.1.2.1.5.1.0", 
"1.3.6.1.2.1.5.3.0",
"1.3.6.1.2.1.5.14.0",
"1.3.6.1.2.1.5.17.0",
"1.3.6.1.2.1.5.8.0",
"1.3.6.1.2.1.5.22.0" ,"1.3.6.1.2.1.6.15.0",
"1.3.6.1.2.1.6.10.0" ,
"1.3.6.1.2.1.6.11.0",
"1.3.6.1.2.1.6.6.0",
"1.3.6.1.2.1.6.12.0",
"1.3.6.1.2.1.6.9.0",
"1.3.6.1.2.1.6.8.0",
"1.3.6.1.2.1.6.5.0",
"1.3.6.1.2.1.7.1.0",
"1.3.6.1.2.1.7.4.0",
"1.3.6.1.2.1.7.3.0",
"1.3.6.1.2.1.7.3.0"]
dataset = []
se = []
loop = True
vol = 0


import mysql.connector
mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="ctiproject")
mycursor = mydb.cursor()

while loop:
    for x in row:
        rec = list(get("127.0.0.1", [x], hlapi.CommunityData("ctiproject")).values())
        rec = str(rec[0])
        se.append(rec) # print(se)
    print(se)
    rec = [[]]
    rec[0] = se
    predictions1 = classifier.predict(rec)
    print(predictions1)
    se = []
    # sql = "INSERT INTO `traffic` (`1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `24`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32`, `33`, `34`, `predict`) VALUES  ( %s, %s, %s, %s, %s, %s, %s, %s, %s , %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    # val = (rec[0][0], rec[0][1], rec[0][2], rec[0][3], rec[0][4], rec[0][5], rec[0][6], rec[0][7], rec[0][8], rec[0][9], rec[0][10], rec[0][11], rec[0][12], rec[0][13], rec[0][14], rec[0][15], rec[0][16], rec[0][17], rec[0][18], rec[0][19], rec[0][20], rec[0][21], rec[0][22], rec[0][23], rec[0][24], rec[0][25], rec[0][26], rec[0][27], rec[0][28], rec[0][29], rec[0][30], rec[0][31], rec[0][32], rec[0][33], predictions1)
    # mycursor.execute(sql, val)
    # mydb.commit()
    time.sleep(15)

# INSERT INTO `traffic` (`id`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `24`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32`, `33`, `34`, `time`, `predict`) VALUES (NULL, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '');




        
        
        
        
        
        
        
    
        
        
        

























































# # Create the scikit-learn logistic regression classifier:
# classifier = LogisticRegression()

# # Train the classifier:
# classifier.fit(training_inputs, training_outputs)

# predictions = classifier.predict(testing_inputs)

# # print(predictions)


# accuracy = metrics.f1_score(testing_outputs, predictions)
# print("LogisticRegression:", accuracy)


# # conmatrix = pd.DataFrame(
# #     metrics.confusion_matrix(testing_outputs, prediction),
# #     index=[['actual', 'actual'], ['pos', 'neg']],
# #     columns=[['predicted', 'predicted'], ['pos', 'neg']]
# # )

# # print(conmatrix)




# # Compute the predictions



# accuracy = metrics.f1_score(testing_outputs, predictions1)
# print("DecisionTree:", accuracy)


# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(training_inputs, training_outputs)

# predicter = classifier.predict(testing_inputs)
# accurater = metrics.f1_score(testing_outputs, predicter)
# print("KNN:", accurater)