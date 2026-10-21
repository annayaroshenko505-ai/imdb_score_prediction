# IMDb Score Prediction – Exploratory Data Analysis and Machine Learning

## Projektübersicht

In diesem Projekt wird untersucht, wie gut sich IMDb-Bewertungen von Filmen anhand verschiedener Metadaten vorhersagen lassen.

Ein besonderer Schwerpunkt liegt auf dem Umgang mit fehlenden Werten, schiefen Verteilungen, Ausreißern, seltenen Kategorien und potenzieller Multikollinearität.

**Ziel:** Vorhersage des IMDb-Scores eines Films auf Basis seiner verfügbaren Merkmale und Vergleich verschiedener Regressionsansätze.

## Datensatz

Verwendet wird der Datensatz `movie_metadata.csv` mit Informationen zu Filmen, Schauspielern, Regisseuren, Genres und weiteren Metadaten.

Die verwendeten Merkmale umfassen unter anderem:

- **Filminformationen:** Laufzeit, Erscheinungsjahr, Budget, Einnahmen und Seitenverhältnis
- **Bewertungen und Popularität:** Anzahl der IMDb-Stimmen und Facebook-Likes
- **Besetzung und Produktion:** Facebook-Likes der Schauspieler und des gesamten Casts
- **Kategorische Merkmale:** Genre, Sprache, Produktionsland, Altersfreigabe und Farbformat

Die Zielvariable ist `imdb_score`.

Der Datensatz enthält fehlende Werte, stark rechtsschiefe numerische Variablen sowie seltene Kategorien und potenzielle Ausreißer.

## Technologien

- Python
- Pandas und NumPy
- Matplotlib und Seaborn
- Scikit-learn
- Jupyter Notebook

## Projektworkflow

### 1. Explorative Datenanalyse (EDA)

Zunächst werden die Struktur, die Verteilungen und die Zusammenhänge innerhalb des Datensatzes untersucht.

Die Analyse umfasst:

- Untersuchung der Datensatzstruktur und Datentypen
- Analyse fehlender Werte
- Deskriptive Statistiken
- Visualisierung der IMDb-Score-Verteilung
- Korrelationsanalyse numerischer Features
- Scatterplots zur Untersuchung von Zusammenhängen zwischen Features und IMDb-Score
- Korrelations-Heatmaps
- Boxplots zur Identifikation potenzieller Ausreißer
- Berechnung des prozentualen Anteils an Ausreißern anhand der IQR-Methode

### 2. Data Cleaning und Feature Selection

Vor der Modellierung werden doppelte Einträge entfernt und ausgewählte Merkmale ausgeschlossen.

Die Spalten `movie_title`, `plot_keywords` und `movie_imdb_link` werden entfernt, da sie für das Basismodell nicht direkt benötigt werden.

Die Namen von Regisseuren und Schauspielern werden ebenfalls ausgeschlossen, um die hohe Dimensionalität durch zahlreiche individuelle Kategorien zu begrenzen.

Ziel ist ein übersichtlicheres Feature-Set, das sich für den Vergleich verschiedener Regressionsmodelle eignet.

### 3. Aufteilung in Train, Validation und Test

Der bereinigte Datensatz wird in drei Teilmengen aufgeteilt:

| Training | 60 % 
| Validation | 20 % 
| Test | 20 % 

Die Preprocessing-Parameter werden anhand der Trainingsdaten bestimmt und anschließend auf die Validation- und Testdaten angewendet.

### 4. Umgang mit seltenen Kategorien

Kategorische Merkmale wie `country`, `language` und `content_rating` enthalten teilweise sehr seltene Ausprägungen.

Um die Anzahl wenig repräsentierter Kategorien zu reduzieren und potenzielles Overfitting zu begrenzen, werden Kategorien mit weniger als 50 Beobachtungen im Trainingsdatensatz zu `Other` zusammengefasst.

Die Gruppierung basiert ausschließlich auf den Trainingsdaten.

### 5. Genre-Encoding

Die Spalte `genres` enthält mehrere Genres pro Film, die durch das Zeichen `|` getrennt sind.

Da ein Film mehreren Genres gleichzeitig angehören kann, wird `MultiLabelBinarizer` für das Multi-Hot-Encoding verwendet.

Dadurch wird jedes Genre als eigenes binäres Feature dargestellt. Ein Film kann somit für mehrere Genre-Features gleichzeitig den Wert 1 erhalten.

Der Encoder wird ausschließlich auf den Trainingsdaten angepasst und anschließend auf Validation und Test angewendet.

### 6. Behandlung fehlender Werte

Fehlende Werte werden abhängig vom Datentyp behandelt:

- **Numerische Features:** Imputation mit dem Median
- **Kategorische Features:** Imputation mit dem häufigsten Wert (Modus)

Die Imputationswerte werden ausschließlich aus dem Trainingsdatensatz berechnet, um Data Leakage durch die Verwendung von Informationen aus Validation oder Test zu vermeiden.

### 7. Logarithmische Transformation

Stark rechtsschiefe numerische Features werden anhand ihrer Schiefe (Skewness) identifiziert.

Features mit einer Skewness größer als 1 werden zunächst als potenzielle Kandidaten ausgewählt. Anschließend wird geprüft, ob eine logarithmische Transformation inhaltlich sinnvoll ist.

Nichtnegative Features werden bei Bedarf mit `np.log1p()` transformiert. Bestimmte strukturelle oder begrenzte Variablen, beispielsweise `duration`, `aspect_ratio` und `facenumber_in_poster`, werden von dieser Transformation ausgeschlossen.

Ziel ist es, extreme Werte weniger dominant zu machen und die Modellierung stark schiefer Verteilungen zu verbessern.

### 8. One-Hot-Encoding und Feature Alignment

Die verbleibenden kategorischen Merkmale werden mithilfe von One-Hot-Encoding in numerische Features umgewandelt.

### 9. Feature Scaling

Für die numerischen Features wird `RobustScaler` verwendet.

Im Gegensatz zur Standardisierung mit `StandardScaler` basiert `RobustScaler` auf dem Median und dem Interquartilsabstand (IQR). Dadurch reagiert er weniger empfindlich auf extreme Werte.

Der Scaler wird ausschließlich auf den numerischen Features des Trainingsdatensatzes trainiert. Die gleichen Parameter werden auf Validation und Test angewendet.

Die binären Features aus dem Genre-Encoding und One-Hot-Encoding bleiben unverändert.

## Modellierung

Es werden vier Regressionsansätze betrachtet.

### 1. Dummy Regressor – Baseline

Der `DummyRegressor` mit `strategy="mean"` sagt für jeden Film den durchschnittlichen IMDb-Score der Trainingsdaten voraus.

Dieses Modell dient als einfache Referenz, um zu überprüfen, ob die komplexeren Modelle einen tatsächlichen Vorhersagegewinn erzielen.

### 2. Lineare Regression

Die lineare Regression modelliert den Zusammenhang zwischen den Features und dem IMDb-Score mithilfe einer linearen Funktion.

Sie dient als einfaches interpretierbares Modell und als Ausgangspunkt für den Vergleich mit regularisierten und nichtlinearen Verfahren.

### 3. Ridge Regression

Die Ridge Regression erweitert die lineare Regression um eine L2-Regularisierung.

Durch die Bestrafung großer Koeffizienten soll das Modell stabiler werden und weniger stark auf einzelne Features reagieren.

Dies ist insbesondere bei vielen korrelierten Features und einer hohen Anzahl von Dummy-Variablen relevant.

Für das Modell wird zunächst `alpha=1.0` verwendet.

Zusätzlich werden die Koeffizienten untersucht und visualisiert, um Features mit besonders großen positiven oder negativen Modellkoeffizienten zu identifizieren.

### 4. Random Forest Regressor

Als nichtlinearer Ansatz wird ein `RandomForestRegressor` trainiert.

Die Hyperparameter werden mithilfe von `GridSearchCV` und einer fünffachen Kreuzvalidierung optimiert.

Untersucht werden folgende Parameter:

| Hyperparameter | Werte |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

Als Optimierungskriterium dient \(R^2\).

Nach der Hyperparametersuche wird das beste Modell ausgewählt und auf den Validation- und Testdaten bewertet.

Zusätzlich werden die Feature Importances analysiert, um die für den Random Forest wichtigsten Merkmale zu identifizieren.

## Evaluation

Die Modelle werden anhand von drei Regressionsmetriken bewertet:

| Metrik | Interpretation |
|---|---|
| MAE – Mean Absolute Error | Mittlere absolute Abweichung zwischen Vorhersage und tatsächlichem IMDb-Score |
| MSE – Mean Squared Error | Mittlerer quadratischer Vorhersagefehler; große Fehler werden stärker gewichtet |
| R² – Bestimmtheitsmaß | Misst, wie gut das Modell die Varianz der Zielvariable erklärt |

Für MAE und MSE sind kleinere Werte besser. Für \(R^2\) sind höhere Werte im Allgemeinen besser.

Die Validation-Daten dienen der Bewertung und dem Vergleich der Modelle. Die Testdaten werden für die abschließende Einschätzung der Generalisierungsleistung verwendet.

## Analyse der Modellkoeffizienten und Feature Importances

Die Ridge-Koeffizienten ermöglichen eine Untersuchung der Richtung und Stärke der modellierten Zusammenhänge.

Positive Koeffizienten stehen bei ansonsten konstanten Features für höhere vorhergesagte IMDb-Scores, negative Koeffizienten für niedrigere Vorhersagen.

Die Feature Importances des Random Forest zeigen, welche Merkmale für die Vorhersagen des trainierten Modells besonders relevant sind.

## Mögliche Einschränkungen

Die Ergebnisse hängen von der Qualität und Zusammensetzung des Datensatzes sowie von der Auswahl und Aufbereitung der Features ab.

Besonders wichtig ist die zeitliche Verfügbarkeit der Merkmale: Variablen wie Einnahmen oder die Anzahl der IMDb-Stimmen sind möglicherweise erst nach Veröffentlichung eines Films bekannt. Werden sie zur Vorhersage eines zukünftigen IMDb-Scores verwendet, kann dies zu einer unrealistischen Einschätzung der Vorhersageleistung führen.

Daher muss die Feature-Auswahl zum konkreten Vorhersageszenario passen.

Außerdem sollte die Hyperparameteroptimierung ausschließlich auf den Trainingsdaten erfolgen. Die Testdaten sollten nicht zur Modellauswahl verwendet werden.

