# Preprocesing 
El archivo ```ml-helpper.py```contiene todas las funciones utiles de la asignatura de Aprendizaje Automático

El paquete ```Preprocessing``` tiene estructurados cada uno de las funciones de ```ml-helpper.py``` de la siguiente manera:

```Preprocessing\datachecking.py```
- missig_values
- outliers

```Preprocessing\discretization.py```
- discretize

```Preprocessing\featureselection.py```
- select_features

```Preprocessing\featureextraction.py```
- get_PCA
- get_score_2_features_subset

```Preprocessing\unbalnced.py```

**OverSampling**
- Borderline_SMOTE
- SMOTE
- ADASYN
- Random

**UnderSampling**
- NearMiss
- ENN
- Tomeks_Link
- Random


## Ejemplos

### ```Preprocessing\datachecking.py```

Missing Values: Borrar NaN

```python
import pandas as pd
from Preprocessing.datachecking import DataChecking
nan_df = pd.read_csv("notebooks/nan_data.csv")

# Borrar Missing Values. Modifica directamente la variable nan_df
print(len(nan_df.values[:,:]))
DataChecking.missig_values(nan_df, action="remove", debug=True)
print(len(nan_df.values[:,:]))

```

Missing Values: Imputar NaN

```python
import pandas as pd
from Preprocessing.datachecking import DataChecking
nan_df = pd.read_csv("notebooks/nan_data.csv")

# Imputar Missing Values. Modifica directamente la variable nan_df
print(len(nan_df.values[:,:]))
DataChecking.missig_values(nan_df, action="impute", debug=True)
print(len(nan_df.values[:,:]))
```

Outliers: Tratar individualmente

```python
import pandas as pd
from Preprocessing.datachecking import DataChecking

df = pd.read_csv("notebooks/iris_data.csv")

# Posicion de todos los outliers. Modifica directamente la variable df
outlier_positions = DataChecking.outliers(df, action="individual", debug=True, remove=True)
print(outlier_positions)
df.describe()
# Podemos ver como faltan 4 valores. Corresponden a los outliers borrados
```

Outliers: Tratar colectivamente

```python
import pandas as pd
from Preprocessing.datachecking import DataChecking

df = pd.read_csv("notebooks/iris_data.csv")

# Posicion de todos los outliers. Modifica directamente la variable df
outlier_positions = DataChecking.outliers(df, action="colective", debug=True, remove=True)
print(outlier_positions)
df.describe()
# Podemos ver como faltan 15 valores. Corresponden a los outliers borrados
```

Outliers: Tratar en paralelo

```python
import pandas as pd
from Preprocessing.datachecking import DataChecking

df = pd.read_csv("notebooks/iris_data.csv")

# Posicion de todos los outliers. Modifica directamente la variable df
outlier_positions = DataChecking.outliers(df, action="parallel", debug=True, remove=True)
print(outlier_positions)
df.describe()
# Podemos ver como faltan 19 valores (4 individuales + 15 colectivos). Corresponden a los outliers borrados
```


### ```Preprocessing\datachecking.py```

Discretizamos X

```python
import pandas as pd
from Preprocessing.discretization import Discretization
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("notebooks/iris_data.csv")

k3frequency = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

# Discretizar con Bins = 3 y Strategy = quantiel
df_discretized = Discretization.discretize(df, k3frequency)
print(df_discretized)
```

### ```Preprocessing\featureselection.py```

Seleccionar las features mas relevantes. Internamente usa ```ExtraTreesClassifier```  (https://scikit-learn.org/stable/modules/feature_selection.html)    

```python
import pandas as pd
from Preprocessing.featureselection import FeatureSelection

import pandas as pd
df = pd.read_csv("../notebooks/iris_data.csv")

selected_features = FeatureSelection.select_features(df)
print(selected_features)
# las features seleccionadas han sido: ['petal_length', 'petal_width']
```


### ```Preprocessing\featureextraction.py```

Obtener un DataFrame que capture el 0.95 de variabilidad usando PCA

```python
import pandas as pd
from Preprocessing.featureextraction import FeatureExtraction

df = pd.read_csv("notebooks/Thyroids.csv")
df_pca = FeatureExtraction.get_PCA(df, variance=0.95)
print(df_pca)
# Utiliza 4 principal compoents
```

Obtener un DataFrame con 40 principal components
```python
import pandas as pd
from Preprocessing.featureextraction import FeatureExtraction

df = pd.read_csv("notebooks/Thyroids.csv")
df_pca = FeatureExtraction.get_PCA(df, variance=40)
print(df_pca)
# Utiliza 40 principal compoents
```

Obtener TODOS los score por cada 2 variables en el DataFrame.

**Atencion**: La funcion ```get_score_2_features_subset``` utiliza ```Util.automatic_scoring```. ```automatic_scoring``` utiliza RandomForest. (Cambiar ```automatic_scoring``` si se quiere utilizar otro clasificador)
```python
import pandas as pd
from Preprocessing.featureselection import FeatureSelection

df = pd.read_csv("notebooks/iris_data.csv")
scores = FeatureSelection.get_score_2_features_subset(df)
print(scores)
```

### ```Preprocessing\unbalnced.py```

#### OverSampling

Hace oversampling con SMOTE

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_oversampled = Unbalanced.OverSampling.SMOTE(df)
```

Hace oversampling con ADASYN
```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_oversampled = Unbalanced.OverSampling.ADASYN(df)
```

Hace oversampling con Random
```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_oversampled = Unbalanced.OverSampling.Random(df)
```

Hace oversampling con Borderline SMOTE version 1
```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
#Borderline_SMOTE tiene 2 versiones: variant=1 y varaint=2
df_ovesampled = Unbalanced.OverSampling.Borderline_SMOTE(df, variant=1)
```




#### UnderSampling

Hacer undersampling con NearMiss 1 

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_undersampled = Unbalanced.UnderSampling.NearMiss(df, variant=1)
```

Hacer undersampling con NearMiss 2 

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_oversdf_undersampledampled = Unbalanced.UnderSampling.NearMiss(df, variant=2)
```

Hacer undersampling con ENN 

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_undersampled = Unbalanced.UnderSampling.ENN(df)
```

Hacer undersampling con Tomeks Link 

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_undersampled = Unbalanced.UnderSampling.Tomeks_Link(df)
```

Hacer undersampling con Random

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_undersampled = Unbalanced.UnderSampling.Random(df)
```

#### Mixto

Hacer oversampling (SMOTE) + undersampling (Tomeks Link)

```python
import pandas as pd
from Preprocessing.unbalnced import Unbalanced

df = pd.read_csv("notebooks/Thyroids.csv")
df_ovesampled = Unbalanced.OverSampling.SMOTE(df)
df_ovesampled_undersampled = Unbalanced.UnderSampling.Tomeks_Link(df_ovesampled)
```