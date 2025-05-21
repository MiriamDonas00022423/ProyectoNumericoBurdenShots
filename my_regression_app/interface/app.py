import gradio as gr
import numpy as np
import joblib
import sys
import os

# ✅ Añadir la ruta a la carpeta 'models' para que Python pueda encontrar los scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

# ✅ Importar las clases desde los archivos de entrenamiento
from train_linear_model import MultipleLinearRegressionGradientDescent
from train_logistic_model import LogisticRegressionGradientDescent

# ✅ Cargar modelos entrenados desde notebook/
linear_model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "notebook","linear_regression", "linear_model.pkl"))
logistic_model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "notebook", "logistic_regression","logistic_model.pkl"))
scaler_lin = joblib.load(os.path.join(os.path.dirname(__file__), "..", "notebook", "linear_regression", "scaler_linear.pkl"))
feature_order = joblib.load(os.path.join(os.path.dirname(__file__), "..", "notebook", "linear_regression", "feature_order.pkl"))
scaler_log = joblib.load(os.path.join(os.path.dirname(__file__), "..", "notebook", "logistic_regression", "scaler_logistic.pkl"))


country_cols = ['Country_Afghanistan','Country_Albania', 'Country_Algeria', 'Country_Angola', 'Country_Antigua and Barbuda', 'Country_Argentina', 'Country_Armenia', 'Country_Australia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Bahamas', 'Country_Bahrain', 'Country_Bangladesh', 'Country_Barbados', 'Country_Belarus', 'Country_Belgium', 'Country_Belize', 'Country_Benin', 'Country_Bhutan', 'Country_Bolivia (Plurinational State of)', 'Country_Bosnia and Herzegovina', 'Country_Botswana', 'Country_Brazil', 'Country_Brunei Darussalam', 'Country_Bulgaria', 'Country_Burkina Faso', 'Country_Burundi', 'Country_Cabo Verde', 'Country_Cambodia', 'Country_Cameroon', 'Country_Canada', 'Country_Central African Republic', 'Country_Chad', 'Country_Chile', 'Country_China', 'Country_Colombia', 'Country_Comoros', 'Country_Congo', 'Country_Costa Rica', 'Country_Croatia', 'Country_Cuba', 'Country_Cyprus', 'Country_Czechia', "Country_Côte d'Ivoire", "Country_Democratic People's Republic of Korea", 'Country_Democratic Republic of the Congo', 'Country_Denmark', 'Country_Djibouti', 'Country_Dominican Republic', 'Country_Ecuador', 'Country_Egypt', 'Country_El Salvador', 'Country_Equatorial Guinea', 'Country_Eritrea', 'Country_Estonia', 'Country_Ethiopia', 'Country_Fiji', 'Country_Finland', 'Country_France', 'Country_Gabon', 'Country_Gambia', 'Country_Georgia', 'Country_Germany', 'Country_Ghana', 'Country_Greece', 'Country_Grenada', 'Country_Guatemala', 'Country_Guinea', 'Country_Guinea-Bissau', 'Country_Guyana', 'Country_Haiti', 'Country_Honduras', 'Country_Hungary', 'Country_Iceland', 'Country_India', 'Country_Indonesia', 'Country_Iran (Islamic Republic of)', 'Country_Iraq', 'Country_Ireland', 'Country_Israel', 'Country_Italy', 'Country_Jamaica', 'Country_Japan', 'Country_Jordan', 'Country_Kazakhstan', 'Country_Kenya', 'Country_Kiribati', 'Country_Kuwait', 'Country_Kyrgyzstan', "Country_Lao People's Democratic Republic", 'Country_Latvia', 'Country_Lebanon', 'Country_Lesotho', 'Country_Liberia', 'Country_Libya', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia', 'Country_Maldives', 'Country_Mali', 'Country_Malta', 'Country_Mauritania', 'Country_Mauritius', 'Country_Mexico', 'Country_Micronesia (Federated States of)', 'Country_Mongolia', 'Country_Montenegro', 'Country_Morocco', 'Country_Mozambique', 'Country_Myanmar', 'Country_Namibia', 'Country_Nepal', 'Country_Netherlands', 'Country_New Zealand', 'Country_Nicaragua', 'Country_Niger', 'Country_Nigeria', 'Country_Norway', 'Country_Oman', 'Country_Pakistan', 'Country_Panama', 'Country_Papua New Guinea', 'Country_Paraguay', 'Country_Peru', 'Country_Philippines', 'Country_Poland', 'Country_Portugal', 'Country_Qatar', 'Country_Republic of Korea', 'Country_Republic of Moldova', 'Country_Romania', 'Country_Russian Federation', 'Country_Rwanda', 'Country_Saint Lucia', 'Country_Saint Vincent and the Grenadines', 'Country_Samoa', 'Country_Sao Tome and Principe', 'Country_Saudi Arabia', 'Country_Senegal', 'Country_Serbia', 'Country_Seychelles', 'Country_Sierra Leone', 'Country_Singapore', 'Country_Slovakia', 'Country_Slovenia', 'Country_Solomon Islands', 'Country_Somalia', 'Country_South Africa', 'Country_South Sudan', 'Country_Spain', 'Country_Sri Lanka', 'Country_Sudan', 'Country_Suriname', 'Country_Swaziland', 'Country_Sweden', 'Country_Switzerland', 'Country_Syrian Arab Republic', 'Country_Tajikistan', 'Country_Thailand', 'Country_The former Yugoslav republic of Macedonia', 'Country_Timor-Leste', 'Country_Togo', 'Country_Tonga', 'Country_Trinidad and Tobago', 'Country_Tunisia', 'Country_Turkey', 'Country_Turkmenistan', 'Country_Uganda', 'Country_Ukraine', 'Country_United Arab Emirates', 'Country_United Kingdom of Great Britain and Northern Ireland', 'Country_United Republic of Tanzania', 'Country_United States of America', 'Country_Uruguay', 'Country_Uzbekistan', 'Country_Vanuatu', 'Country_Venezuela (Bolivarian Republic of)', 'Country_Viet Nam', 'Country_Yemen', 'Country_Zambia', 'Country_Zimbabwe']


def predecir_lineal(pais, *args):
    base_names = [
    "Year", "Adult Mortality", "percentage expenditure", "Measles", "under-five deaths",
    "Polio", "Total expenditure", "HIV/AIDS", "thinness 10-19 years",  
    "Income composition of resources", "Schooling", "Status_Developing"
    ]

    base_inputs = list(args)
    base_dict = dict(zip(base_names, base_inputs))

    # Codificar país
    country_dict = {col: 0 for col in country_cols}
    if pais in country_dict:
        country_dict[pais] = 1

    # Combinar todo en un solo dict
    full_input_dict = {**base_dict, **country_dict}

    # Reordenar según feature_order
    x_ordered = np.array([[full_input_dict[col] for col in feature_order]])

    # Escalar y predecir
    x_scaled = scaler_lin.transform(x_ordered)
    pred = linear_model.predict(x_scaled)

    return f"📈 Esperanza de vida estimada: {pred[0]:.2f} años"

def predecir_logistica(*args):
    input_names = ['age', 'bmi', 'smoke_formerly smoked', 'smoke_never smoked', 'smoke_smokes', 'avg_glucose_level']
    input_values = list(args)
    input_dict = dict(zip(input_names, input_values))

    # Reordenar y convertir a DataFrame
    import pandas as pd
    X_df = pd.DataFrame([input_dict], columns=input_names)

    # Escalar
    X_scaled = scaler_log.transform(X_df)

    # Predecir
    pred = logistic_model.predict(X_scaled)
    return "🟥 Riesgo de stroke: ALTO" if pred[0] == 1 else "🟩 Riesgo de stroke: BAJO"

# --- Interfaz con pantallas independientes ---
with gr.Blocks(title="App de Regresión", theme=gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="pink",
    neutral_hue="gray"
), css="""
.custom-header {
    color: #d14d8b !important;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
}
.custom-button {
    background: #f8c8dc !important;
    color: #333 !important;
    border: 1px solid #f8c8dc !important;
}
.custom-button:hover {
    background: #f5b5d1 !important;
}
.gr-box {
    border: 1px solid #f8c8dc !important;
    border-radius: 8px !important;
}
""") as demo:
    current_page = gr.State("home")

    # --- Pantalla de inicio ---
    with gr.Column(visible=True) as pantalla_inicio:
        gr.Markdown("# Bienvenido a la App del Proyecto #1 de Análisis Numérico", 
                   elem_classes=["custom-header"])
        gr.Markdown("En este proyecto tenemos dos modelos de machine learning entrenados por medio de gradiente descendiente, uno de regresión lineal y otro de regresión logistica")
        with gr.Row():
            btn_go_lineal = gr.Button("Regresión Lineal", 
                                    elem_classes=["custom-button"])
            btn_go_log = gr.Button("Regresión Logística", 
                                 elem_classes=["custom-button"])

    # --- Pantalla Lineal ---
    with gr.Column(visible=False) as pantalla_lineal:
        gr.Markdown("## Regresión Lineal", 
                   elem_classes=["custom-header"])

        with gr.Row():
            with gr.Column(scale=3):
                country_dropdown = gr.Dropdown(choices=country_cols, label="País")
                year_input = gr.Number(label="Year")
                adult_mortality = gr.Number(label="Adult Mortality")
                expenditure_perce = gr.Number(label="Percentage expenditure")
                measles = gr.Number(label="Measles")
                five_deaths = gr.Number(label="Under-five deaths")
                polio_input = gr.Number(label="Polio")
                total_exp = gr.Number(label="Total expenditure")
                hiv_input = gr.Number(label="HIV/AIDS")
                thiness_input = gr.Number(label="Thinness 10-19 years")
                compos_input = gr.Number(label="Income composition of resources")
                schooling = gr.Number(label="Schooling")
                dev_input = gr.Number(label="Status Developing (1=SÍ, 0=NO)")

                base_inputs = [
                    year_input, adult_mortality, expenditure_perce, measles, five_deaths,
                    polio_input, total_exp, hiv_input, thiness_input,
                    compos_input, schooling, dev_input
                ]

            with gr.Column(scale=2):
                btn_lineal = gr.Button("Predecir", 
                                     elem_classes=["custom-button"])
                out_lineal = gr.Textbox(label="Resultado", lines=3)
                btn_back1 = gr.Button("Volver al inicio", 
                                    elem_classes=["custom-button"])

    # --- Pantalla Logística ---
    with gr.Column(visible=False) as pantalla_log:
        gr.Markdown("## Regresión Logística", 
                   elem_classes=["custom-header"])

        with gr.Row():
            with gr.Column(scale=3):
                age_input = gr.Number(label="Age")
                bmi_input = gr.Number(label="BMI")
                former_smoke = gr.Number(label="Smoke formerly smoked (1=SÍ, 0=NO)")
                never_smoke = gr.Number(label="Smoke never smoked (1=SÍ, 0=NO)")
                smokes = gr.Number(label="Smoke smokes (1=SÍ, 0=NO)")
                glucose_input = gr.Number(label="Avg glucose level")

                inputs_log = [
                    age_input, bmi_input, former_smoke, never_smoke, smokes, glucose_input
                ]

            with gr.Column(scale=2):
                btn_log = gr.Button("Predecir", 
                                  elem_classes=["custom-button"])
                out_log = gr.Textbox(label="Resultado", lines=3)
                btn_back2 = gr.Button("Volver al inicio", 
                                    elem_classes=["custom-button"])


    # --- Lógica de navegación ---
    def show_page(page):
        return {
            pantalla_inicio: gr.update(visible=(page == "home")),
            pantalla_lineal: gr.update(visible=(page == "lineal")),
            pantalla_log: gr.update(visible=(page == "log")),
            current_page: page
        }

    btn_go_lineal.click(lambda: show_page("lineal"), outputs=[pantalla_inicio, pantalla_lineal, pantalla_log, current_page])
    btn_go_log.click(lambda: show_page("log"), outputs=[pantalla_inicio, pantalla_lineal, pantalla_log, current_page])
    btn_back1.click(lambda: show_page("home"), outputs=[pantalla_inicio, pantalla_lineal, pantalla_log, current_page])
    btn_back2.click(lambda: show_page("home"), outputs=[pantalla_inicio, pantalla_lineal, pantalla_log, current_page])

    # --- Eventos de predicción ---
    btn_lineal.click(predecir_lineal, inputs=[country_dropdown] + base_inputs, outputs=out_lineal)
    btn_log.click(predecir_logistica, inputs=inputs_log, outputs=out_log)

if __name__ == "__main__":
    demo.launch()

