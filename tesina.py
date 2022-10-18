# ANDREA BALDINELLI
# tesina Machine Learning and Data Analysis

import matplotlib as mpl
from matplotlib import  pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

mpl.use('TkAgg')
pd.set_option('display.expand_frame_repr', False) #per non tagliare il print


# realizzata solo per permettere di mettere il codice in pausa!
def pause():
    input("Premi <ENTER> per continuare ...\n")


# IMPORT DATASET
# DIMENSIONI DATASET: (19158, 14)
data = pd.read_csv('dataset/aug_train.csv')
print(f"\ndimensione dataset: {data.shape}\n")

# visualizzazione preliminare di informazioni del dataset
# features dei campioni del dataset
features = list(set(data.columns) - {'target'})
print("lista features : ",features)

#  experience', 'enrolled_university', 'city', 'major_discipline', 'last_new_job',
# 'company_type', 'training_hours', 'city_development_index', 'enrollee_id',
# 'company_size', 'education_level', 'gender', 'relevent_experience'


# visualizzazione delle prime 10 colonne del dataset
print(data.head(10),"\n")

# da una prima visualizzazione, sembra che ci sia qualche valore a null
sns.heatmap(data.isnull())
plt.title("valori nulli presenti nel dataset")
plt.show()

# si può osservare come ci siano motli valori nulli all'interno della colonna "genere",
# anche nella colonna "major discipline" e anche nelle colonne relative alla dimensione
# dell'ultima azienda in cui i candidati hanno lavorato ("company size" e "company type")
# prima di decidere il da farsi, dobbiamo valutare bene cosa fare

# visualizzazione "tipi" dei dati
print("\ndomini delle features:\n",data.dtypes)

# enrollee_id                 int64
# city                       object
# city_development_index    float64
# gender                     object
# relevent_experience        object
# enrolled_university        object
# education_level            object
# major_discipline           object
# experience                 object
# company_size               object
# company_type               object
# last_new_job               object
# training_hours              int64
# target                    float64
# dtype: object

# sono molte features object, vanno tutte gestite
pause()


########################################################################################################################
# OSSERVIAMO LA LABEL 'TARGET'
# visualizzazione distribuzione delle due classi all'interno del dataset
target = data['target']
target_counts = target.value_counts(dropna=False)
print(target_counts)
target_counts.plot.pie(labels=target_counts.index.tolist())
plt.title("distribuzione delle classi \nclassi sbilanciate")
plt.show()


# dataset molto sbilanciato! in percentuale
class_0_occurrency = sum(data['target'] == 0.0)
print("\npersone che NON sono in cerca di un lavoro: ",class_0_occurrency/len(data['target']))
print("persone che sono in cerca di un lavoro: ",1 - class_0_occurrency/len(data['target']))

pause()
########################################################################################################################
# iniziammo con la feature 'CITY_DEVELOPMENT_INDEX', un coefficiente che descrive il livello
# di sviluppo della città (indicata dalla feat city), indicato con un numero compreso tra 0 e 1
cities_index = data['city_development_index'].value_counts(dropna=False)

sns.displot(x=data['city_development_index'], kind='kde')
plt.title("distribuzione della feature 'city_development_index'")
plt.grid()
plt.show()

print("\n", cities_index)
print("\nelementi nulli: ",sum(data['city_development_index'].isna())) # non ha valori nulli! molto bene!!


# una distribuzione non "polarizzata"

# quantizzazione in quartile, in modo da poter plottare i valori
city_development_index = pd.qcut(data['city_development_index'],q=4,labels=False)
print("numero di occorrenze: ",city_development_index.value_counts())
sns.countplot(x=city_development_index,hue=target)
plt.title("distribuzione della feature 'city_develop_index' in funzione della feature target")
plt.show()

# sono int64, quindi non male come cosa!

# LA FEAT 'city' la possiamo anche scartare, in quanto fortemente correlata con 'city_develop_index' e per questo mi porta ad
# avere standard error elevato nella stima dei parametri
data.drop(columns=['city'], axis=1,inplace=True)
pause()

########################################################################################################################
# FEATURE "GENDER". Come già visionato all'inizio, presenta valori nulli
# print("valori nulli colonna 'gender': ", sum(data['gender'].isna()))
# # sono 4508 valori nulli, su un totale di 19158 sample. Può essere dovuto da errore, oppure potrebbe essere disponibile la voce "non specificato"
# però, potremmo mettere tutto sotto Other

# data['gender'] = data['gender'].fillna(value='Other')
gender_counts = data['gender'].value_counts(dropna=False)
gender_counts.plot.pie(labels=gender_counts.index.tolist())
plt.title("distribuzione dei generi dei candidati all'interno del dataset di training")
plt.show()

# gestiamo i valori a NaN
data['gender'] = data['gender'].fillna(value="no_specified")
# data['gender_is_specified'] = data[(data['gender'] == 'male') | (data['gender'] == 'female')]

# vediamo come influenza l'uscita la conoscenza del gender
sns.countplot(x=data["gender"],hue=target)
plt.title("distribuzione del target in funzione del genere dei candidati")
plt.show()

# mapping_value = { True: 1,
#                   False: 0}
# data['gender_is_specified'] = data['gender_is_specified'].map(mapping_value)


# per renderlo operativamente applicabile, dobbiamo renderla operativamente utilizzabile
# dobbiamo dummyzzare la feature!
dummy1 = pd.get_dummies(data=data['gender'], prefix="gender",drop_first=True)
print(dummy1.columns)
# aggiungiamo le nuove feature al dataframe

data['gender_is_specified'] = 1 - dummy1['gender_no_specified']
# rimuoviamo genere!
data.drop(columns=['gender'],axis=1,inplace=True)
print("lista colonne dataframe: ", data.columns)

pause()

########################################################################################################################
# andiamo a valutare la feature 'RELEVENT_EXPERIENCE', indica se il candidato ha esperienza, o meno, pertinente con il lavoro di data scientist
relevent_experience_counts = data['relevent_experience'].value_counts(dropna=False)
print("valori nulli della colonna 'relevant_experience': ", sum(data['relevent_experience'].isna()))

relevent_experience_counts.plot.pie(labels=data['relevent_experience'].unique())
plt.title("distribuzione feature 'relevant_experience'")
plt.show()

sns.countplot(x= data['relevent_experience'], hue=target)
plt.title("distruibuzione feat_relevent_experience in funzione delle classi d'uscita")
plt.show()

# poichè questa feat non è numerica, deve essere resa tale dummyzzando
dummy2 = pd.get_dummies(data= data['relevent_experience'], drop_first=True)

# per poter avere la feat "in positivo"
data['relevant_experience'] = 1 - dummy2['No relevent experience']

# PICCOLO CHECK
# print("correlazione fra esperienza nel settore, del candidato e la probabilità di essere assunto",data['relevant_experience'].corr(target))

data.drop(columns=['relevent_experience'],axis=1,inplace=True)
print(data.columns) # DEBUG

pause()


########################################################################################################################
# andiamo ad approfondire la feature 'ENROLLED_UNIVERSITY'
# descrive il tipo di corso universitario in cui il candidato è iscritto

enrolled_university_counts = data['enrolled_university'].value_counts(dropna=False)
enrolled_university_counts.plot.pie(labels=enrolled_university_counts.index.tolist())
plt.title("distribuzione dei valori di 'enrolled_university'")
plt.show()
# ho 386 valori a nan, non saprei come fillarli, alla peggio li rimuovo. Però ho letto la possibilità di usare Knn come filler
# di NAN! Il punto è che, questa feat è molto sbilanciata verso No_enrollment!
# prima rendo la feat numerica

# per andare a feare un mapping
map_values = { 'no_enrollment' : 1.0, 'Full time course': 2.0, 'Part time course':3.0}
data['enrolled_university'] = data['enrolled_university'].map(map_values)

# in una maniera non proprio corretta, "fillo" i valori a nan con il valore medio della feat.
data['enrolled_university'] = data.enrolled_university.fillna(value=np.round(data['enrolled_university'].mean()))

enrolled_university_counts = data['enrolled_university'].value_counts(dropna=False)
print("distribuzione feat 'enrolled_university': ",enrolled_university_counts)



sns.countplot(x= data['enrolled_university'], hue= target)
plt.title("distribuzione della feat enrolled_uni in funzione del target")
plt.show()

# questa feature rimane in sospeso, perchè l'essersi iscritto in università non mi dice se poi l'ha portata a termine o meno!
# quindi attenzione

# REVERSE MAPPING dei valori
mapp = {1.0 :'no_enrollment',
        2.0 :'Full time course',
        3.0 :'Part time course'}

data['enrolled_university'] = data['enrolled_university'].map(mapp)

enrolled_university_counts = data['enrolled_university'].value_counts(dropna=False)
enrolled_university_counts.plot.pie(labels=enrolled_university_counts.index.tolist())
plt.title("distribuzione feat 'enrolled_uni' nel Dataset, dopo il fix dei valori nan")
plt.show()

# dummyzzazione della feature, fatta alla fine
# non droppo enrolled_uni, la lascio e la droppo in seguito

pause()
########################################################################################################################
# approfondiamo la variabile 'EDUCATION LEVEL'
# come al solito andiamo ad approfondire la distribuzione della feat all'interno del dataset!

edu_level_counts = data['education_level'].value_counts(dropna=False)
print("distribuzione della feat education_level del candidato: ",edu_level_counts)

edu_level_counts.plot.pie(labels=edu_level_counts.index.tolist())
plt.title("distribuzione della feat 'education_level'")
plt.show()

#voglio vedere la distribuzione con 'relevant_experience'
sns.countplot(x=data['education_level'], hue=data['relevant_experience'])
plt.title("distribuzione 'edu_level' in funzione della feat 'relevant_experience'")
plt.show()

# come un po' mi aspettavo, abbiamo che per gli "studiati" hanno tutti molta più esperienza nel campo"

sns.countplot(x=data['education_level'], hue=data['enrolled_university'])
plt.title("distribuzione 'edu_level' in funzione della feat 'enrolled_university'")
plt.show()
# questo invece mi mette ancora più dubbi su quello che significa enrolled university
# mantiene senso solo se vuol dire chi "ATTUALMENTE" FREQUENTA L'UNIVERSITÀ

# poichè ci sono persone con la licenza elementare che sono iscritti all'università
cleaning_education_level = data.loc[data['education_level'] == 'Primary School','enrolled_university']
print(cleaning_education_level.value_counts(dropna=False))

print(data.shape)
data = data[~((data['education_level']=='Primary School') & (data['enrolled_university'] != 'no_enrollment'))]
target = data['target']
print(data.shape)

sns.countplot(x=data['education_level'], hue=target)
plt.title("distribuzione 'edu_level' in funzione del target")
plt.show()
# in proporzione ho più laureati con target!

# è una variabile sì categotica, però ordinale! Il grado scolastico va reso crescente!!!
edu_level_mapping = {'Phd': 4, 'Masters': 3, 'Graduate': 2, 'High School': 1, 'Primary School': 0}
data['education_level'] = data['education_level'].map(edu_level_mapping)
# posso renderla poi dummy comunque!


# # a questo punto,PER ORA, DROPPO LE COLONNE, ALTRIMENTI (ANCHE QUI) SOSTITUISCO CON LA MEDIA
data['education_level'] = data['education_level'].fillna(value=np.round(data['education_level'].mean(skipna=True)))
print(data.shape)


print("correlazione 'edu_level' con 'relevant_experience'",data['education_level'].corr(data['relevant_experience']))

sns.countplot(x=data['education_level'], hue=target)
plt.title("distribuzione 'edu_level' in funzione di una assunzione o meno")
plt.show()


pause()
########################################################################################################################
# controlliamo la feature 'MAJOR_DISCIPLINE'. Ovvero la materia di "competenza"
print("valori assunti da 'major_discipline': ",data['major_discipline'].unique())
print("distribuzione valori di 'major_discipline': ",data['major_discipline'].value_counts(dropna=False))


#graficamente
major_discipline_counts = data['major_discipline'].value_counts(dropna=False)
major_discipline_counts.plot.pie(labels=major_discipline_counts.index.tolist())
plt.title("distribuzione delle 'major_discipline'")
plt.show()

# in un curriculum data science, è comunque avere come major discipline STEM, rispetto alle altre
# abbiamo tanti valori a nan, questo però è quasi corretto, perchè chi ha un titolo di studio 'High school' o 'primary_school'
# non hanno un campo di specializzazione!
# major discipline è a nan in quei casi in cui il candidato non è un laureato et simila
# posso fillare quei valori nulli con 'No major'!

sns.countplot(x=data.loc[data['major_discipline'].isna(),'education_level'])
plt.title("distribuzione dei valori nulli di major_discipline\nin funzione del livello scolastico")
plt.show()
# come ci aspettavamo, i campioni avente come valore 'High school' o 'Primary_school' non hanno una specializzazione in qualche
# disciplina!

high_school_discipline_nan = sum(data.loc[(data['education_level'] == 1) | (data['education_level']== 0),"major_discipline"].isna())
print('numero di campioni nulli in major discipline \n noto che stiamo osservando persone con "diploma" o licenza "elementare": ',high_school_discipline_nan),
# quasi l'intero totale!
print('numero di campioni con major discipline a 1: ', data[(data['education_level']==1) | (data['education_level']==0)].shape)
# tutti i campioni a 1 hanno questa feat a nan!
print(data.loc[(data['education_level'] ==1) | (data['education_level'] == 0)].shape)

#vediamo distribuzione valori nulli, che conferma quanto detto
major_discipline_distribution = data.loc[data['major_discipline'].isna(), 'education_level'].value_counts()
print('\ndistribuzione valori nulli', major_discipline_distribution)
#2325 sono valori a nan semplicemente perchè il loro grado scolastico (high_school e primary_school) non li specializza!

# quindi, sostituiamo quei 2325 valori nan con 'No major_discipline'. Gestione dei valori a nan
data.loc[data['education_level']==1,'major_discipline'] = data.loc[data['education_level']==1 ,'major_discipline'].fillna(value='No Major')
data.loc[data['education_level']==0,'major_discipline'] = data.loc[data['education_level']==0,'major_discipline'].fillna(value='No Major')
# i restanti valori a nan
# faccio il fill con il valore Other
data['major_discipline'] = data['major_discipline'].fillna(value='Other')


target = data['target']
print("dopo la rimozione",data.shape)

# vediamo se abbiamo "sistemato i null"
major_discipline_counts = data['major_discipline'].value_counts(dropna=False)
major_discipline_counts.plot.pie(labels=major_discipline_counts.index.tolist())
plt.title("distribuzione delle 'major_discipline' dopo il fix dei nan")
plt.show()
# pause


sns.countplot(x=data['major_discipline'], hue=target)
plt.title("distribuzione delle discipline in funzione delle classi in uscita")
plt.show()

sns.countplot(x=data['major_discipline'], hue=data['relevant_experience'])
plt.title("distribuzione delle discipline in funzione della relevant experience")
plt.show()
# pause



# poichè il focus del tutto è assumere persone con curriulum data science, posso  dummyzzare così
dummy3 = pd.get_dummies(data=data['major_discipline'],prefix='competence',drop_first=True)
data = pd.concat([data, dummy3['competence_STEM'],dummy3['competence_No Major']], axis=1)
data.drop(columns=['major_discipline'],axis=1,inplace=True)

print("\ncontrollo aggiunta nuova feature andato a buon fine\n",data.columns)



pause()

# ######################################################################################################################
# osserviamo la feature 'EXPERIENCE', Indica l'esperienza totale del candidato, espressa in anni

print("distribuzione dei valori della feature experience",data['experience'].value_counts(dropna=False))
# 53 valori nulli
data['experience'].value_counts().plot.pie(labels= data['experience'].value_counts().index.tolist())
plt.title("distribuzione dei valori della feature experience")
plt.show()

# o vuol dire che sono persone che non hanno mai lavorato, oppure sono appena usciti dal percorso di studio
# posso andare a verificare la distribuzione dei valori nulli

# verifichiamo come si distribuiscono in funzione del titolo di studio
nan_experience_distribution = data.loc[data['experience'].isnull(), 'education_level']
print("distribuzione valori nulli di experience, in funzione del loro grado di educazione scolastica",nan_experience_distribution.value_counts())

nan_experience_distribution.value_counts().plot.pie(labels=nan_experience_distribution.value_counts().index.tolist())
plt.title("distribuzione valori nulli di experience in funzione del livello di istruzione")
plt.show()

# RESOCONTO
# 2.0    32
# 3.0    13
# 1.0     4
# 4.0     3
# 0.0     1

# anche in questo caso, ad avere nessun tipo di esperienza, sono quelli che (presumo) siano da poco usciti dal percorso scolastico
# ergo posso considerarli come apprentice, ovvero esperienza minore di un anno

#aggiungo anche la condizione che
nan_experience = data.loc[(data['experience'].isna()) & (data['competence_STEM']==1),'relevant_experience'].value_counts()
print(nan_experience)
nan_experience.plot.pie(labels=nan_experience.index.tolist())
plt.title("distribuzione dei valori nulli nella feat experience\nin funzione della feature 'relevant_experience'")
plt.show()


# quelli 'senza esperienza' sono comunque persone con competenza e con una specializzazione STEM! Questo, assunzione mia,
# vuol dire che molto probabilmente avranno anche la feat last_new_job a 1!!
nan_experience_2 = data.loc[(data['experience'].isna()) & (data['competence_STEM']==1) & (data['relevant_experience']==1),'last_new_job'].value_counts()
print(nan_experience_2)
nan_experience_2.plot.pie(labels=nan_experience_2.index.tolist())
plt.title("distribuzione dei valori nulli di experience in funzione della relevant_experience a 1\ncompetence_STEM a 1, in funzione di last_new_job")
plt.show()

# gestione dei valori a nan
data['experience'] = data['experience'].fillna(value='<1')


# <1 apprenticev
# 1-5 junior
# 6-10 intermediate
# 11-15 pro
# 16->20 senior

def experience_trheshold(experience):

        if (experience == '<1'):
            return 'apprentice'
        elif experience == '>20':
            return 'senior'
        elif int(experience) >= 1 and int(experience) <= 10:
            return 'junior'
        elif int(experience) >= 11 and int(experience) <= 19:
            return 'pro'
        else: # experience == 20
            return 'senior'


#applico funzione ad experience
data['experience'] = data.apply(lambda x:experience_trheshold(x['experience']),axis=1)
print(data.experience.value_counts())
# pause()

sns.countplot(x=data['experience'], hue=target)
plt.title("distribuzione esperienza candidati in funzione del target")
# plt.show()
# come è giusto che sia, gli apprendisti sono "propensi a cambiare lavoro"

#voglio vedere come si modella questa feat con has experience
sns.countplot(x=data['experience'], hue=data['relevant_experience'])
plt.title("distribuzione esperienza del candidato")
plt.show()

# come era lecito aspettarsi, tranne gli "apprentice" sono molti di più quelli che hanno esperienza (grosso boom dei junior)
# invece per gli "apprentice", sono molti di più i candidati senza esperienza "relativa"


# invece di dummyzzare, posso dire che i "nomi" che ho dato all'esperienza comunque hanno una certa gerarchia (ordinamento)
# exp_map = { 'apprentice' : 1,
#             'junior' : 2,
#             'pro'   : 3,
#             'senior' : 4}

#data['experience'] = data['experience'].map(exp_map)
pause()

########################################################################################################################
# andiamo a sistemare la feature 'LAST_NEW_JOB'

print("distribuzione valori nel dataset: ",data.last_new_job.value_counts(dropna=False))
data.last_new_job.value_counts(dropna=False).plot.pie(labels=data.last_new_job.value_counts(dropna=False).index.tolist())
plt.title("distribuzione valori last_new_job nel dataset")
plt.show()

# ci sono 381 valori a null, andiamo a vedere come si "modellano" in funzione dell'experience
# verifichiamo se questo valore, coincide con persone "apprentice", ovvero senza troppa esperienza relativa
sns.countplot(x=data['relevant_experience'], hue=data['last_new_job'])
plt.title("distribuzione valori secondo l'esperienza 'relativa' degli iscritti")
plt.show()

# voglio verificare anche la relazione in funzione all'esperienza dei candidati
sns.countplot(x=data['experience'], hue=data['last_new_job'].isna())
plt.title("distribuzione valori nulli secondo l'esperienza 'lavorativa' degli iscritti")
plt.show()


dist_nan_experience = data.loc[data['last_new_job'].isna(), 'experience'].value_counts()
print(dist_nan_experience)

# pause
# junior        260
# apprentice     54
# pro            38
# senior         29

# pause
# in questo caso, per sistemare i valori nulli, prima di tutto trasformo in numerica la variabile e poi mi calcolo la media
# di last_new_job in funzione del tipo di esperienza

# trasformazione variabile in numerica, per poter calcolare la media
num_mapping = { '>4': 5.0,
                '4': 4.0,
                '3': 3.0,
                '2': 2.0,
                '1': 1.0,
                'never': 0.0}

data['last_new_job'] = data['last_new_job'].map(num_mapping)

unique_experience = ["apprentice","junior","pro","senior"]
relative_mean = {}
nan_handling = data[~(data['last_new_job'].isna())]

for key in unique_experience:
    relative_mean[key] = np.round(nan_handling.loc[nan_handling['experience'] == key ,'last_new_job'].mean(skipna=True))
    # calcolo la media in questo modo, riempo un dizionario in cui
    # key = experience,
    # value = la media.

print(relative_mean)

def apply_mean(cols):
    # funzione che mi serve per sostituire i valori nan con la media di 'last_new_job' in funzione relativa all'esperienza del candidato
    experience = cols[0]
    last_new_job = cols[1]
    if pd.isnull(last_new_job):
        if experience == 'no_experience':
            return  relative_mean['no_experience']
        elif experience == 'apprentice':
            return relative_mean['apprentice']
        elif experience == 'junior':
            return relative_mean['junior']
        elif experience == 'pro':
            return relative_mean['pro']
        else:
            return relative_mean['senior']
    else:
        return last_new_job

data['last_new_job'] = data[['experience','last_new_job']].apply(apply_mean,axis=1)


# gestiti i valori nulli
sns.countplot(hue=data['last_new_job'],x=data.loc[data['relevant_experience']==1,'experience'])
plt.title("distribuzione valori last_new_job secondo l'esperienza 'lavorativa' degli iscritti\n(noto che hanno esperienza 'relativa')")
plt.grid()
plt.show()


sns.countplot(x=data['last_new_job'], hue=target)
plt.title("distribuzione degli anni dall'ultimo lavoro (prima dell'attuale), in funzione delle classi target")
plt.show()
# C'è da dire che non è molto discrimante, visto che più o meno hanno sempre la stessa classificazione!


pause()
########################################################################################################################
# # # osserviamo la feature 'COMPANY_TYPE'
# andiamo a vedere la distribuzione della feat 'company_type'

company_type_counts = data['company_type'].value_counts(dropna=False)
print('distribuzione company_type : ', company_type_counts)

# 5000 e passa valori nulli
company_type_counts.plot.pie(labels=company_type_counts.index.tolist())
plt.title("distribuzione dei valori di 'company_type' nel dataset")
plt.show()

# gestiamo i molti valori nulli
# verifico questi valori nulli che relazione hanno con last new job
last_job_never = data[data['company_type'].isnull()]
print(last_job_never.shape)
sns.countplot(x=last_job_never['last_new_job'])
plt.title("distribuzione valori nulli di company type\nin funziione di last_new_job")
plt.show()


# posso incrociare i dati con la feat 'company_size'
print('company size dei campioni con company_type nullo: ',data.loc[data['company_type'].isnull(),'company_size'].value_counts(dropna=False))
pause()

# vado a vedere come influisce questa feat nell'uscita
sns.countplot(x=data.loc[~(data['company_type'].isna()), 'company_type'], hue=data['target'])
plt.title("distribuzione della feat 'company_type', in funzione dell'uscita")
plt.show()


# gestione dei valori a nan, fill con la media
map_company_type = { 'Pvt Ltd' :1, 'Funded Startup':2, 'Public Sector':3, 'Early Stage Startup':4, 'NGO':5,'Other':6 }

data['company_type'] = data['company_type'].map(map_company_type)
data['company_type'] = data['company_type'].fillna(value= np.round(data['company_type'].mean(skipna=True)))
# oppure potevo inserire all'interno di 'Other'

# reverse mapping
reverse_mapping = { 1:'Pvt Ltd',
    2:'Funded Startup',
    3:'Public Sector',
    4:'Early Stage Startup',
    5:'NGO',
    6:'Other'
}

pause()
########################################################################################################################
# gestiamo 'COMPANY_SIZE'
company_size_counts = data['company_size'].value_counts(dropna=False)
company_size_counts.plot.pie(labels=company_size_counts.index.tolist())
plt.title("distribuzione 'company_size'")
plt.show()

print('distribuzione valori company_size: ',company_size_counts)

# AVREI PREFERITO USARE TECNICHE COME IL "RESAMPLING VIA KNNImputer", MA MI DA PROBLEMI.
company_size_map = {
    '50-99':3,
    '100-500':4,
    '10000+':8,
    '10/49' :2,
    '1000-4999' :6,
    '<10' :1,
    '500-999':5,
    '5000-9999': 7
}

data['company_size'] = data['company_size'].map(company_size_map)
data['company_size'] = data['company_size'].fillna(value= np.round(data['company_size'].mean(skipna=True)))

sns.countplot(x= data['company_size'], hue=data['target'])
plt.title("distribuzione 'company_size' in funzione del target")
plt.show()

pause()
########################################################################################################################
# gestiamo 'TRAINING_HOURS', per vedere qualche high leverage point....
print("descrizione feature 'training_hours': ",data['training_hours'].describe())

training_hours = data['training_hours'].value_counts(dropna=False)
print("distribuzione feature 'training_hours': ",training_hours)

sns.displot(x=data['training_hours'],kind='kde')
plt.title("distribuzione feature 'training_hours' ")
plt.grid()
plt.show()

sns.scatterplot(x=data['training_hours'], y=data['training_hours'],hue=data['target'])
plt.title("verifica presenza High Leverage Points nella feat 'training_hours")
plt.show()

training_hour = data[['training_hours','target']]
training_hour['training_hours'] = pd.qcut(training_hour['training_hours'],q=4,labels=False)
sns.countplot(x=training_hour['training_hours'], hue=training_hour['target'])


plt.title("distribuzione feat 'training_hours' in funzione del target")
plt.show()

pause()
########################################################################################################################
# ultime pulizie

# perchè non sono più state rimosse queste features?
# data.drop(columns='company_type',axis=1,inplace=True)
# data.drop(columns='company_size',axis=1,inplace=True)


data.drop(columns='enrollee_id',axis=1,inplace=True)

# dummyzzo la feature 'experience'
dummy4 = pd.get_dummies(data=data['experience'],drop_first=True,prefix='experience')
data = pd.concat([data,dummy4],axis=1)
data.drop(columns='experience',axis=1,inplace=True)

# dummyzziamo la feature enrolled university
dummy5 = pd.get_dummies(data=data['enrolled_university'],drop_first=True)
data = pd.concat([data,dummy5],axis=1)
data.drop(columns='enrolled_university',axis=1,inplace=True)

# rendo dummy la variabile company_type
# prima la ritrasformo in una feature categorica nominale
data['company_type'] = data['company_type'].map(reverse_mapping)
dummy6 = pd.get_dummies(data['company_type'], drop_first=True, prefix='company')
data = pd.concat([data,dummy6], axis=1)
data.drop(columns='company_type', axis=1, inplace=True)


# print(data.dtypes)
plt.show()

sns.heatmap(data.corr(),annot=True)
plt.title("correlazione features Dataset")
plt.show()


# rimozione del target
target = data['target']
data.drop(columns='target',axis=1,inplace=True)
print(data.shape)
print(data.columns)

# droppo 'Funded_Startup', visto l'alta correlazione con un'altra feature
data.drop(columns='company_Funded Startup',axis=1, inplace=True)

print("dopo c'è addestramento, proseguire con catuela!!!!")
pause()
# ######################################################################################################################
## ADDESTRAMENTO MODELLO

# resampling con tecninca SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE(random_state=2)
data, target = oversample.fit_resample(data, target)
print(data.shape)

from sklearn.model_selection import train_test_split
#split training e test set
X_train,X_test,Y_train,Y_test = train_test_split(data,target,test_size=0.20,random_state=42,shuffle=True)

m,p = X_train.shape
# feature dummy
X_train = np.concatenate((np.ones((m,1)), X_train), axis=1)

mt,pt = X_test.shape
# feature dummy
X_test = np.concatenate((np.ones((mt,1)),X_test),axis=1)

## standardizziamo le features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#modelli da provare
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import  Pipeline



# per poter avere una pipeline
class DummyEstimator(BaseEstimator):
    def fit(self): pass

    def score(self): pass

# Create a pipeline
pipe = Pipeline([('clf', DummyEstimator())])  # Placeholder Estimator

search_space = [{'clf': [LogisticRegression()],
                 'clf__penalty': ['l2','none'],
                 'clf__C': np.logspace(0, 5, 20)},
                {'clf': [MLPClassifier(max_iter=1000)],
                 'clf__hidden_layer_sizes':[(50,), (100,),(50,50),(100, 100)],
                'clf__alpha':[0.0001, 0.001,0.01, 0.1, 1,10,100],
                'clf__solver':['sgd','adam'],
                'clf__activation':['tanh','relu']}
                # NON HO POTUTO PROVARE SVC IN QUANTO TROPPO "LENTO" NEL FITTING, il mio pc si "rifiutava"
                # {
                #  'clf': [svm.SVC()],
                #  'clf__C': [0.001,0.01,0.1,1,10,30],
                #  'clf__kernel': ['linear','rbf','poly'],
                #  'clf__gamma': [0.01, 0.1, 1, 10, 100],
                #  'clf__degree': [3, 5, 7]
                #  }
                ]


# Create grid search CV
model = GridSearchCV(pipe, search_space,n_jobs=-1,verbose=10,scoring='f1',cv=5)
model.fit(X_train,Y_train)

print("migliori parametri risultanti: ",model.best_params_)

# verifica bontà dell'algoritmo
Y_pred_train = model.predict(X_train)
Y_pred = model.predict(X_test)


# score modello
print("classification report sul training: \n\n",classification_report(Y_train,Y_pred_train))
print("classification report sul test: \n\n",classification_report(Y_test,Y_pred))

# auc score, solo per completezza
from sklearn.metrics import roc_auc_score,confusion_matrix
print("metrica auc score sul training: ",roc_auc_score(Y_train, Y_pred_train, average='macro'))
print("metrica auc score sul test set: ",roc_auc_score(Y_test, Y_pred, average='macro'))

# CONFUSION MATRIX costruita sui campioni di test
sns.heatmap(confusion_matrix(Y_test,Y_pred),annot=True)
plt.title("confusion matrix")
plt.show()


# RISULTATI
# PROVA 1
# mlp classifier, SENZA LE FEATURE 'COMPANY_TYPE' e 'COMPANY_SIZE'

# #             precision    recall  f1-score   support
# #
# #          0.0       0.76      0.83      0.79     11181
# #          1.0       0.81      0.74      0.77     11230
# #
# #     accuracy                           0.78     22411
# #    macro avg       0.79      0.78      0.78     22411
# # weighted avg       0.79      0.78      0.78     22411
# #
# #               precision    recall  f1-score   support
# #
# #          0.0       0.75      0.82      0.78      2826
# #          1.0       0.80      0.72      0.76      2777
# #
# #     accuracy                           0.77      5603
# #    macro avg       0.77      0.77      0.77      5603
# # weighted avg       0.77      0.77      0.77      5603



# PROVA 2, CON LE FEAT COMPANY SIZE E TYPE "FILLATE" IN MANIERA NON PROPRIAMENTE CORRETTA
# elapsed: 114.4min finished
# {'clf': MLPClassifier(activation='tanh', alpha=0.1, hidden_layer_sizes=(100, 100),max_iter=1000), 'clf__solver': 'adam'}

# TRAIN
#               precision    recall  f1-score   support
#          0.0       0.83      0.82      0.82     11181
#          1.0       0.83      0.83      0.83     11230
#
#     accuracy                           0.83     22411
#    macro avg       0.83      0.83      0.83     22411
# weighted avg       0.83      0.83      0.83     22411


# TEST
#               precision    recall  f1-score   support
#          0.0       0.82      0.81      0.81      2826
#          1.0       0.81      0.81      0.81      2777
#
#     accuracy                           0.81      5603
#    macro avg       0.81      0.81      0.81      5603
# weighted avg       0.81      0.81      0.81      5603


# AUC SCORES
# TRAIN: 0.8255285390247082
# TEST: 0.812617278060787
