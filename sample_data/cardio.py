import streamlit as st
import streamlit.components.v1 as components
#from streamlit_pandas_profiling import st_profile_report
#from pandas_profiling import ProfileReport 
import pandas as pd
from PIL import Image
import time  
import os
import plotly.express as px 
import numpy as np
import sklearn
import cufflinks as cf
import plotly.graph_objects as go
cf.go_offline()


#template html
html_temp = """
<div style="background-color:Black; border:2px solid black;border - radius:20px"><p style="color:white;font-size:44px;padding:10px">heart attack by statistical learning</p></div>
	"""
st.set_page_config(html_temp, layout='wide')
st.text('Tags:|health, health conditions heart, conditions diseases|💼 Usability 10.0, License ⚖')
#set image
image1 = Image.open("Cardiovascularr.jpg")
st.image(image1, caption='Cardiovascular Disease',use_column_width=True)

#set image in sidebar
image = Image.open("Human-cardiovascular-system.jpg")
st.sidebar.image(image, caption='Cardiovascular Disease',use_column_width=True)

#print title
st.sidebar.text('The purpose of this document is to create a machine learning model that will be able to predict cardiovascular disease based on a set of data')


st.sidebar.markdown("""
# Context 
A retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa.
There are roughly two controls per case of coronary heart disease. Many of the coronary heart disease
positive men have undergone blood pressure reduction treatment and other programs to reduce their risk
factors after their coronary heart disease event. In some cases the measurements were made after these
treatments. These data are taken from a larger dataset, described in Rousseauw et al, 1983, South African
Medical Journal.
""" )

st.sidebar.markdown("""
# MetaData
""" )

select_content = st.sidebar.selectbox("choice one option and show informations",('No select option','Content/Data Description','Published by','Sources','Inspiration'))
if select_content == 'Content/Data Description':
  st.subheader('Content/Data Description')
  st.write('A data frame with 462 observations on the following 10 variables.')
  st.text('sbp ====> systolic blood pressure')
  st.text('tobacco ====> cumulative tobacco (kg)')
  st.text('ldl ====> low density lipoprotein cholesterol')
  st.text('adiposity ====> a numeric vector')
  st.text('famhist ====> family history of heart disease, a factor with levels "Absent" and "Present"')
  st.text('typea ====> type-A behavior')
  st.text('obesity ====> a numeric vector')
  st.text('alcohol ====> current alcohol consumption')
  st.text('age ====> age at onset')
  st.text('chd ====> response, coronary heart disease')
elif select_content == 'Published by':
  st.write('This dataset is published by SAheart: South African Hearth Disease Data .')
elif select_content == 'Sources':
  st.subheader('Sources')
  st.write('Rousseauw, J., du Plessis, J., Benade, A., Jordaan, P., Kotze, J. and Ferreira, J. (1983). Coronary risk factor screening in three rural communities, South African Medical Journal 64: 430–436.ElemStatLearn, R-Package')
elif select_content == 'Inspiration':
  st.subheader('Inspiration')
  st.write('The objective of this document is to show “empirically” the potential interest of thedimension reduction in Machine Learning with set algorithms.')

st.sidebar.markdown("""
# Parameter of Data
""" )

def main():

  data = pd.read_csv("cardiovascular.csv")

  st.sidebar.text('Basic information')
  with st.sidebar.beta_expander('Select here',expanded=False):
      # ler o arquivo
      if st.checkbox("Show the Dataset "):
        show_data = st.progress((0))
        with st.spinner(f"⏳ Loading ..."):
            time.sleep(1)
        for percent_completo in range(100):
            time.sleep(0.01)
            show_data.progress(percent_completo+1)
        st.success('The database have been successfully loaded ✔️')
        st.write(data)
      
      #visualizar o valor minimo em cada coluna
      elif st.checkbox('max value of columns'):
        with st.empty():
            for seconds in range(2):
              st.write(f"⏳ {seconds} seconds have passed")
              time.sleep(1)
            st.write(f"✔️ {seconds}  seconde over!")
        st.write(data.style.highlight_max(axis=0))
        st.info('😀 Move the row below the Dataframe to view the maximum values ​​in each column')
  
      #visualizar o valor minimo em cada coluna
      elif st.checkbox('min value of columns'):
        with st.empty():
            for seconds in range(2):
              st.write(f"⏳ {seconds} seconds have passed")
              time.sleep(1)
            st.write(f"✔️ {seconds}  seconde over!")
        st.write(data.style.highlight_min(axis=0))
        st.info('😀 Move the row below the Dataframe to view the minimum values ​​in each column')

      # visualizar a quantidade de linhas e colunas
      elif st.checkbox('number of rows and columns'):
        st.subheader(data.shape)
        st.code('462 rows and 11 columns')

      # visualizar o nome das variaveis
      elif st.checkbox('List name of variables'):
         st.code(data.columns)
      
      #tipo de dados
      elif st.checkbox('Data type'):
         st.code(data.dtypes)
      
      #unique value in column(')
      elif st.checkbox('Unique value in column chd'):
        st.text(data['chd'].unique())

      elif st.checkbox('Evaluating for Missing Data 🏸'):
        st.table(data.isnull().sum())
        st.info('No missing data')
  

  st.sidebar.markdown("""
  # Data Cleaning
  """ )
  #replae value of columns
  new_data = data.replace({'famhist' : { 'Absent' : 0, 'Present' : 1 }})
  
  # set index
  new_data = new_data.set_index('ind')
  
  # radio
  genre = st.sidebar.radio(
     "",
     ('No select','New Database', 'New Database type'))
  if genre == 'New Database':
    values = st.slider('Select the amount of line you want to view the data',0, 462)
    components.html("<p style ='color:blue;'>In this table we perform some manipulations and replace the unique values ​​that are in the column famhist with (Absent = 0 and Present = 1)'</p>")
    st.write(new_data.head(values))

  elif genre == 'New Database type':
    new_data = new_data.astype('float')
    st.table(new_data.dtypes)


  st.sidebar.markdown("""
  # Statistical information
  """ )
  from sklearn import preprocessing 
  select_estatictic = st.sidebar.selectbox("choice one option and show informations",('','Statistic of Data','Statistic of column chd'))
  if select_estatictic == 'Statistic of Data':
    st.subheader('Statistic of the Data')
    st.write(new_data.describe())
    st.subheader("Statistic of variable preditive 'chd' ")
    st.write(new_data["chd"].describe())
    st.subheader("re-escala of data")
    scaler = preprocessing.StandardScaler().fit (new_data[["chd"]])
    st.write(scaler)
    new_data["chd"] = scaler.transform(new_data[["chd"]])
    st.write(new_data["chd"].describe())


  elif select_estatictic == 'Statistic of column chd':
    chd_height = new_data.chd.value_counts()/new_data.shape[0]
    st.subheader("there is no coronary disease = 0, there is coronary disease = 1")
    st.code(chd_height)

  st.sidebar.markdown("""
  # Preview
  """ )

  

  select_grafic = st.sidebar.radio(
      "",
      ('No select','bar chart','Scatter chart','Violin plot'))

  if select_grafic == 'Profile Report of the Data':
    profile = ProfileReport(data, html={'style':{'full_width':True}})
    st_profile_report(profile)
    
  elif select_grafic == 'bar chart':
      select_value = st.selectbox("select from the list one of the variables you want to view",('sbp', 'tobacco', 'ldl', 'adiposity','famhist', 'typea', 'obesity', 'alcohol','age','chd'))
      if select_value == select_value: 
        fig1 = px.histogram(data, x=select_value)
      if st.button('Plot the bar'):
        st.plotly_chart(fig1) 
          
        
  elif select_grafic == ("Scatter chart"):
    
      c1,c2 = st.beta_columns(2)
        

      with c1:
        with st.beta_expander('x_label'):
            #all_columns = new_data.columns.tolist()
          option_grafic3 = st.selectbox("select x_label from the list one of the variables you want to view",('','chd'))
          if option_grafic3 == option_grafic3:
            option_grafic3=option_grafic3

      with c2:
        with st.beta_expander('y_label'):
          #all_columns = new_data.columns.tolist()
          option_grafic4 = st.selectbox("select y_label from the list one of the variables you want to view",('','sbp', 'tobacco',  'adiposity', 'typea', 'obesity', 'alcohol','age','ldl','famhist'))
          if option_grafic4 == option_grafic4:
            option_grafic4=option_grafic4       
        
      if st.button('Plot the bar'):
        fig3 = px.scatter(data ,x= option_grafic3 , y =option_grafic4, height=500 )
        st.plotly_chart(fig3)

      st.header("Data variables with the predictive variable'chd'")

      c1,c2 = st.beta_columns(2)

      with c1:
        with st.beta_expander('X_label'):
          option_grafic7 = st.selectbox("select X_label from the list one of the variables you want to view",('','sbp', 'tobacco',  'adiposity', 'typea', 'obesity', 'alcohol','age','ldl','famhist'))
          if option_grafic7 == option_grafic7:
            option_grafic7=option_grafic7

      with c2:
        with st.beta_expander('Y_label'):
          option_grafic8 = st.selectbox("select Y_label from the list one of the variables you want to view",('','sbp', 'tobacco',  'adiposity', 'typea', 'obesity', 'alcohol','age','ldl','famhist'))
          if option_grafic8 == option_grafic8:
            option_grafic8=option_grafic8

      if st.button('See Plot '):
        #sns.set_theme(style="whitegrid", palette="muted")
        fig = px.scatter(data, x=option_grafic7, y=option_grafic8, color="chd")
        st.plotly_chart(fig)
        st.write("Result of 'chd'")
        st.code("0 = there is no coronary disease")
        st.code("1 = there is coronary disease")


  elif select_grafic == ("Violin plot"):
    
      c1,c2 = st.beta_columns(2)

      with c1:
        with st.beta_expander('x_label'):
          #all_columns = new_data.columns.tolist()
          option_grafic4 = st.selectbox("select x_label from the list one of the variables you want to view",('','chd'))
          if option_grafic4 == option_grafic4:
            option_grafic4=option_grafic4

      with c2:
        with st.beta_expander('y_label'):
          #all_columns = new_data.columns.tolist()
          option_grafic5 = st.selectbox("select y_label from the list one of the variables you want to view",('','sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity','famhist', 'alcohol','age'))
          if option_grafic5 == option_grafic5:
            option_grafic5=option_grafic5     
              
      
      if st.button('Plot the bar'):
        fig5 = px.violin(data ,x =option_grafic4, y =option_grafic5,color = option_grafic4, box=True, points="all", height=500 )
        st.plotly_chart(fig5)

  
  st.sidebar.markdown("""
  # Machine learning models
  """ ) 

  #Separando as variáveis em preditores X e variável resposta Y
  new_data1 = new_data.drop('chd',axis =1)
  x = new_data1 
  y = new_data['chd']

  #Vamos construir os modelos usando 75% da base e testando sua qualidade nos outros 25%
  from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
  from sklearn.model_selection import train_test_split ## essa função ajuda a separar nos dois conjuntos que falamos acima
  x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.25)

  
  with st.sidebar.beta_expander('Select here',expanded=False):
      select_grafic1 = st.radio("",('No select','Models to be tested','logistic regression','Decison tree','knn classification','support vector machine','Linear discirminant analysis','Quadratic discriminant analysis','Random forest classification','xgboost classification'))

  if select_grafic1 == 'Models to be tested':
    st.json({
    'Models to be tested': [
        'logistic regression',
        'Decison tree',
        'knn classification',
        'support vector machine',
        'Linear discirminant analysis',
        'Quadratic discriminant analysis',
        'Random forest classification',
        'xgboost classification'
    ],
    })

  elif select_grafic1 == 'logistic regression':
    # logistic regression
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(solver='lbfgs' , C=1.0, fit_intercept =True , max_iter =100, n_jobs =1, intercept_scaling = 1)
    lg.fit(x_treino,y_treino)
    Logistic_Regression = lg.score(x_teste,y_teste) #calculando accuracy
    y_pred_lg = lg.predict(x_teste) 
    confusion_matrix_lr = confusion_matrix(y_teste, y_pred_lg) # confusion_matrix
    precision_score_lr = precision_score(y_teste, y_pred_lg, average="macro")# precision score
    f1_score_lg = f1_score(y_teste, y_pred_lg, average="macro")
    st.text('Array')
    st.text(y_pred_lg)
    st.text('Accuracy')
    st.write(Logistic_Regression)
    st.text('confusion matrix')
    st.table(confusion_matrix_lr)
    st.text('precision score')
    st.code(precision_score_lr)
    st.text('f1 score')
    st.code(f1_score_lg) 

  elif select_grafic1 == 'Decison tree':
    # Decison tree
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(x_treino, y_treino)
    DecisionTree_Classifier = dtc.score(x_teste,y_teste)
    y_pred_Dtc = dtc.predict(x_teste)
    confusion_matrix_Dtc = confusion_matrix(y_teste, y_pred_Dtc) # confusion_matrix
    precision_score_Dtc = precision_score(y_teste, y_pred_Dtc, average="macro")# precision score
    f1_score_Dtc = f1_score(y_teste, y_pred_Dtc, average="macro")
    st.text('Array')
    st.text(y_pred_Dtc)
    st.text('Accuracy')
    st.write(DecisionTree_Classifier)
    st.text('confusion matrix')
    st.table(confusion_matrix_Dtc)
    st.text('precision score')
    st.code(precision_score_Dtc)
    st.text('f1 score')
    st.code(f1_score_Dtc) 

  elif select_grafic1 == 'knn classification':
    from sklearn.neighbors import KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=6)
    knc.fit(x_treino,y_treino)
    KNeighbors_Classifier = knc.score(x_teste,y_teste) 
    y_pred_knc = knc.predict(x_teste)
    confusion_matrix_knc = confusion_matrix(y_teste, y_pred_knc) # confusion_matrix
    precision_score_knc = precision_score(y_teste, y_pred_knc, average="macro")# precision score
    f1_score_knc = f1_score(y_teste, y_pred_knc, average="macro")
    st.text('Array')
    st.text(y_pred_knc) 
    st.text('Accuracy')
    st.write(KNeighbors_Classifier)
    st.text('confusion matrix')
    st.table(confusion_matrix_knc) 
    st.text('precision score')
    st.code(precision_score_knc)
    st.text('f1 score')
    st.code(f1_score_knc) 

  elif select_grafic1 == 'support vector machine':
    from sklearn.svm import LinearSVC
    lsvc = LinearSVC(random_state=0, tol=1e-5)
    lsvc.fit(x_treino, y_treino)
    Linear_SVC = lsvc.score(x_teste,y_teste) 
    y_pred_lsvc = lsvc.predict(x_teste)
    confusion_matrix_lsvc = confusion_matrix(y_teste, y_pred_lsvc) # confusion_matrix
    precision_score_lsvc = precision_score(y_teste, y_pred_lsvc, average="macro")# precision score
    f1_score_lsvc = f1_score(y_teste, y_pred_lsvc, average="macro")
    st.text('Array')
    st.text(y_pred_lsvc) 
    st.text('Accuracy')
    st.write(Linear_SVC)
    st.text('confusion matrix')
    st.table(confusion_matrix_lsvc)
    st.text('precision score')
    st.code(precision_score_lsvc)
    st.text('f1 score')
    st.code(f1_score_lsvc) 

  elif select_grafic1 == 'Linear discirminant analysis':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_treino, y_treino)
    LinearDiscriminant_Analysis = lda.score(x_teste,y_teste) 
    y_pred_lda = lda.predict(x_teste)
    confusion_matrix_lda = confusion_matrix(y_teste, y_pred_lda) # confusion_matrix 
    precision_score_lda = precision_score(y_teste, y_pred_lda, average="macro")# precision score
    f1_score_lda = f1_score(y_teste, y_pred_lda, average="macro")
    st.text('Array')
    st.text(y_pred_lda)
    st.text('Accuracy')
    st.write(LinearDiscriminant_Analysis)
    st.text('confusion matrix')
    st.table(confusion_matrix_lda)
    st.text('precision score')
    st.code(precision_score_lda)
    st.text('f1 score')
    st.code(f1_score_lda) 

  elif select_grafic1 == 'Quadratic discriminant analysis':
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_treino,y_treino)
    Quadratic_Discriminant_Analysis = qda.score(x_teste,y_teste) 
    y_pred_qda = qda.predict(x_teste)
    confusion_matrix_qda = confusion_matrix(y_teste, y_pred_qda) # confusion_matrix
    precision_score_qda = precision_score(y_teste, y_pred_qda, average="macro")# precision score
    f1_score_qda = f1_score(y_teste, y_pred_qda, average="macro")
    st.text('Array')
    st.text(y_pred_qda)
    st.text('Accuracy')
    st.write(Quadratic_Discriminant_Analysis)
    st.text('confusion matrix')
    st.table(confusion_matrix_qda)
    st.text('precision score')
    st.code(precision_score_qda)
    st.text('f1 score')
    st.code(f1_score_qda) 

  elif select_grafic1 == 'Random forest classification':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(x_treino,y_treino)
    Random_Forest_Classifier = clf.score(x_teste,y_teste)
    y_pred_RFC = clf.predict(x_teste)
    confusion_matrix_RFC = confusion_matrix(y_teste, y_pred_RFC) # confusion_matrix
    precision_score_RFC = precision_score(y_teste, y_pred_RFC, average="macro")# precision score
    f1_score_RFC = f1_score(y_teste, y_pred_RFC, average="macro")
    st.text('Array')
    st.text(y_pred_RFC) 
    st.text('Accuracy')
    st.write(Random_Forest_Classifier)
    st.text('confusion matrix')
    st.table(confusion_matrix_RFC)
    st.text('precision score')
    st.code(precision_score_RFC)
    st.text('f1 score')
    st.code(f1_score_RFC) 

  elif select_grafic1 == 'xgboost classification':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(x_treino,y_treino)
    Random_Forest_Classifier = clf.score(x_teste,y_teste) 
    y_pred_xgb = clf.predict(x_teste)
    confusion_matrix_xgb = confusion_matrix(y_teste, y_pred_xgb) # confusion_matrix
    precision_score_xgb = precision_score(y_teste, y_pred_xgb, average="macro")# precision score 
    f1_score_xgb = f1_score(y_teste, y_pred_xgb, average="macro")
    #previsao = clf.predict(x_teste[0:2])
    st.text('Array')
    st.text(y_pred_xgb)
    st.subheader('performance of models')
    st.text('Accuracy')
    st.write(Random_Forest_Classifier)
    st.text('confusion matrix')
    st.table(confusion_matrix_xgb)
    st.text('precision score')
    st.code(precision_score_xgb)
    st.text('f1 score')
    st.code(f1_score_xgb)
    #st.write(previsao)
  
  st.sidebar.markdown("""
  # About
  """ ) 
  option_about = st.sidebar.selectbox("developer info",('No_select_info','Name', 'Linkedin','Github'))
  if option_about == ('Name'):
    image_developer = Image.open("developer_app.jpg")
    st.sidebar.image(image_developer, caption='DIETRICH MONTCHO',use_column_width=True)

  elif option_about == ('Linkedin'):
    link_linkedin = 'https://www.linkedin.com/in/dietrich-montcho-b13672121/'
    st.sidebar.markdown(link_linkedin, unsafe_allow_html=True)

  elif option_about == ('Github'):
    link_github = 'https://github.com/Dietrich061991'
    st.sidebar.markdown(link_github, unsafe_allow_html=True)


if __name__ == '__main__':
    	main()

