import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# generar datos

educ = np.random.normal(11, 5, 1000)
e = np.random.normal(0, 4, 1000)
wage = 5 + 2.8 * educ + e

# abrir la página

# título

st.title("Esto es un test")
# header

# incluir gráficos

tab1, tab2, tab3 = st.tabs(['Histograma', 'Scatter', 'Modelo'])

# scatter

with tab1:
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(educ)
    ax[0].set_title("Educación")
    ax[1].hist(wage)
    ax[1].set_title("Salario")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(1, 1)
    ax.scatter(educ, wage, s=10, alpha=0.5)
    st.pyplot(fig)

with tab3:
    X = np.c_[np.ones(len(educ)), educ]
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(wage))

    fig, ax = plt.subplots(1, 1)
    vec_educ = np.linspace(educ.min(), educ.max(), 100)
    ax.scatter(educ, wage, s=10, alpha=0.5)
    ax.plot(vec_educ, betas[0] + betas[1] * vec_educ, color='r')
    st.pyplot(fig)

    educ_st = st.slider("Ingrese sus años de educación: ",
              0, 30)

    if st.button("Predecir"):
        pred = betas[0] + betas[1] * educ_st
        st.write(pred)

