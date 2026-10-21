# IMDb Score Prediction - Movie Data Analysis

## Projektbeschreibung

In diesem Projekt wird untersucht, inwieweit sich IMDb-Bewertungen von Filmen anhand verschiedener Merkmale vorhersagen lassen. Dazu wird ein Regressionsmodell mit Python und Scikit-learn entwickelt.

**Ziel:** Vorhersage des IMDb-Scores eines Films auf Basis seiner verfügbaren Metadaten.

## Datensatz

Das Projekt verwendet den Datensatz `movie_metadata.csv`, der Informationen über Filme enthält, darunter:

- **Bewertung:** `imdb_score`
- **Filminformationen:** Laufzeit, Erscheinungsjahr, Budget und Einnahmen
- **Besetzung und Regie:** Schauspieler und Regisseur
- **Soziale Interaktionen:** Facebook-Likes
- **Kategorische Merkmale:** Genre, Sprache, Land und Altersfreigabe

Der Datensatz enthält fehlende Werte, unterschiedlich skalierte numerische Variablen und teilweise stark schiefe Verteilungen.

## Technologien

- Python
- Pandas und NumPy
- Matplotlib und Seaborn
- Scikit-learn
- Jupyter Notebook

## Projektablauf

### 1. Explorative Datenanalyse (EDA)

Die Analyse umfasst:

- Anzahl der Zeilen und Spalten
- Datentypen und fehlende Werte
- Deskriptive Statistiken
- Verteilung der IMDb-Scores
- Korrelationen zwischen numerischen Features und dem IMDb-Score
- Scatterplots zur Untersuchung möglicher Zusammenhänge
- Korrelationsmatrix zur Analyse der Beziehungen zwischen numerischen Variablen

Zusätzlich wird untersucht, ob eine logarithmische Transformation von `num_voted_users` den Zusammenhang mit dem IMDb-Score besser sichtbar macht.

### 2. Data Cleaning

Vor dem Modelltraining werden die Daten bereinigt.

Die wichtigsten Schritte sind:

- Entfernen doppelter Zeilen
- Entfernen nicht benötigter oder potenziell problematischer Features
- Vereinfachung der Genre-Informationen durch Auswahl des ersten Genres als `main_genre`

Folgende Spalten werden entfernt:

- `movie_title`
- `movie_imdb_link`
- `plot_keywords`
- `actor_1_facebook_likes`
- `actor_2_facebook_likes`
- `actor_3_facebook_likes`
- `facenumber_in_poster`
- `num_user_for_reviews`
- `director_facebook_likes`
- `num_critic_for_reviews`

Die Auswahl berücksichtigt die Relevanz der Features, mögliche Datenqualitätsprobleme und das Risiko von Data Leakage.

### 3. Definition der Zielvariable und Features

Die Zielvariable ist:

`imdb_score`

### 4. Aufteilung der Daten

Der Datensatz wird in drei Teilmengen aufgeteilt:

| Training | 60 % 
| Validation | 20 % 
| Test | 20 % 

### 5. Umgang mit fehlenden Werten

Fehlende Werte werden abhängig vom Datentyp behandelt:

- **Numerische Features:** Ersetzung durch den Median
- **Kategorische Features:** Ersetzung durch den häufigsten Wert (Modus)

Die Imputationswerte werden ausschließlich aus den Trainingsdaten berechnet und anschließend auf die Validation- und Testdaten angewendet. Dadurch wird Data Leakage bei der Imputation vermieden.

### 6. Feature Transformation und Encoding

Um stark rechtsschiefe Verteilungen zu reduzieren, werden folgende Features logarithmisch transformiert:

- `budget`
- `movie_facebook_likes`
- `cast_total_facebook_likes`

Kategorische Variablen werden anschließend mithilfe von One-Hot-Encoding in numerische Features umgewandelt. 

Da die verschiedenen Datensätze nach dem Encoding unterschiedliche Kategorien enthalten können, werden die Validation- und Testdaten an die Spalten des Trainingsdatensatzes angepasst. Fehlende Dummy-Spalten werden mit Nullen ergänzt.

### 7. Feature Scaling

Die numerisch kodierten Features werden mit `StandardScaler` standardisiert.

Der Scaler wird ausschließlich auf den Trainingsdaten trainiert. Die gleichen Skalierungsparameter werden anschließend auf die Validation- und Testdaten angewendet.

### 8. Modellierung

Es werden zwei Modelle betrachtet.

**Baseline – DummyRegressor**

Ein einfaches Referenzmodell sagt für jeden Film den durchschnittlichen IMDb-Score der Trainingsdaten voraus.

Die Baseline dient als Vergleichspunkt, um festzustellen, ob das trainierte Modell bessere Vorhersagen liefert als eine konstante Vorhersage.

**Lineare Regression – LinearRegression**

Als Machine-Learning-Modell wird eine lineare Regression verwendet. Sie soll den IMDb-Score anhand der aufbereiteten Features vorhersagen.

Das Modell wird auf den skalierten Trainingsdaten trainiert.

### 9. Evaluation

Die Modellleistung wird anhand von drei Metriken bewertet:

| Metrik | Bedeutung |
|---|---|
| MSE – Mean Squared Error | Mittlerer quadratischer Vorhersagefehler; kleinere Werte sind besser. |
| MAE – Mean Absolute Error | Mittlere absolute Abweichung zwischen vorhergesagten und tatsächlichen IMDb-Scores; kleinere Werte sind besser. |
| R² – Bestimmtheitsmaß | Beschreibt, wie viel Varianz der Zielvariable durch das Modell erklärt wird. Höhere Werte sind im Allgemeinen besser. |

Die Evaluation erfolgt zunächst auf den Validation-Daten und anschließend auf den Testdaten.

## Projektstruktur

```text
project/
├── movie_metadata.csv
├── imdb_score_prediction.ipynb
└── README.md
```

## Ziel und mögliche Erweiterungen

Mögliche Erweiterungen:

- Vergleich zusätzlicher Regressionsmodelle, beispielsweise Ridge Regression und Random Forest
- Untersuchung nichtlinearer Zusammenhänge
- Vergleich der Modelle anhand derselben Evaluation-Metriken


