import numpy as np


class MultipleLinearRegressionGradientDescent():
    # Constructor: inicializa los parámetros del modelo
    def __init__(self,n_iterations,learning_rate):
        # Vector de pesos
        self.weights = None
        # Término independiente (bias) 
        self.intercept = None
        # Número de iteraciones para entrenar
        self.n_iterations = n_iterations
         # Tasa de aprendizaje  η
        self.lr = learning_rate
         # Lista para guardar el error cuadrático medio en cada iteración
        self.errors = []
    # Función que realiza una actualización de los pesos y el bias
    def _Gradient_Descent(self,n_samples,X,y_preds,y_act):
        # Derivada de la función de costo MSE con respecto a los pesos
        self._Dw = (-1)*(1/n_samples)*np.dot(X.transpose(),(y_act-y_preds))
        # Derivada con respecto al intercepto (término independiente)
        self._Di = (-1)*(1/n_samples)*np.sum(y_act-y_preds)
        # Actualización de los pesos
        self.weights = self.weights-(self.lr*self._Dw)
         # Actualización del intercepto
        self.intercept = self.intercept-(self.lr*self._Di)
    # Método para entrenar el modelo con los datos de entrada X y salidas y
    def fit(self,X,y):
        # Número de ejemplos y características
        n_samples,n_features = X.shape
         # Inicializa pesos en cero
        self.weights = np.zeros(n_features)
        # Inicializa el bias en cero
        self.intercept = 0
        # Ciclo de entrenamiento por número de iteraciones especificado
        for _ in range(self.n_iterations):
             # Predicción
            y_preds = self.intercept+X.dot(self.weights)
             # Error cuadrático medio
            error = np.mean((y-y_preds)**2)
             # Guarda el error para graficar o monitorear convergencia
            self.errors.append(error)
            # Realiza una actualización de los pesos y el bias
            self._Gradient_Descent(n_samples,X,y_preds,y)
    # Método para hacer predicciones con nuevos datos
    def predict(self,x):
        # Convierte a numpy array si no lo es
        if type(x) != 'numpy.ndarray':
            x = np.array(x)
        # Calcula ŷ = Xw + b y devuelve el resultado
        return self.intercept + x.dot(self.weights)
