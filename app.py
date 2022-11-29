import streamlit as st 
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib  
matplotlib.use('Agg')
import seaborn as sns 

from sklearn import model_selection
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier	
from sklearn.naive_bayes  import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

def main():
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.title('📈Seminario de visualización📈')
	st.text('👀Visualizador genérico de datos👀')
	st.text('🚀Construído con Streamlit🚀')
	activities = ["Análisis exploratorio de datos","Gráfico"]

	choice = st.sidebar.selectbox("Seleccionar herramienta",activities)

	if choice == 'Análisis exploratorio de datos' : 
		
		st.subheader("Análisis exploratorio de datos")
		data = st.file_uploader("Cargar Dataset",type = ["csv","txt","xls"])		
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Mostrar instancias y variables"):
				st.write(df.shape)

			if st.checkbox("Mostrar columnas"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Seleccionar columnas a mostrar"):
				all_columns = df.columns.to_list()
				selected_columns = st.multiselect("Seleccionar columnas",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Mostrar sumario"):
				st.write(df.describe())

	elif choice == 'Gráfico' : 
		st.subheader("Visualización de datos")

		data = st.file_uploader("Cargar Dataset",type = ["csv","txt","xls"])		
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		if st.checkbox("Correlación con Seaborn"):
			st.write(sns.heatmap(df.corr(),annot = True))
			st.pyplot()

		if st.checkbox("Gráfico de tarta"):
			all_columns = df.columns.to_list()
			columns_to_plot = st.selectbox("Seleccionar columna",all_columns)
			pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
			st.write(pie_plot)
			st.pyplot()

		all_columns_names = df.columns.to_list()
		type_of_plot = st.selectbox("Seleccionar tipo de gráfico",["area","bar","line","kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

		if st.button("Generar gráfico"):
			st.success("Generando gráfico personalizado {} para {}".format(type_of_plot,selected_columns_names))

			if type_of_plot == 'area':
				collect_data = df[selected_columns_names]
				st.area_chart(collect_data)				
			elif type_of_plot == 'bar':
				collect_data = df[selected_columns_names]
				st.bar_chart(collect_data)

			elif type_of_plot == 'line':
				collect_data = df[selected_columns_names]
				st.line_chart(collect_data)

			elif type_of_plot:
				collect_data = df[selected_columns_names].plot(kind = type_of_plot)
				st.write(collect_data)
				st.pyplot()

if __name__ == '__main__':
	main()

