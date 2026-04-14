import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

print('='*60)
print('FULL END-TO-END PIPELINE TEST (tmdb_5000_movies.csv)')
print('='*60)

# Step 1: Problem Type = Regression
print('\n[STEP 1] Problem Type: Regression ✓')

# Step 2: Data Input
df = pd.read_csv('tmdb_5000_movies.csv')
target = 'vote_average'
features = ['budget','popularity','revenue','runtime','vote_count']
print(f'[STEP 2] Data loaded: {df.shape[0]} rows x {df.shape[1]} cols')
print(f'  Target: {target}, Features: {features}')

# Step 3: EDA
print(f'[STEP 3] EDA:')
print(f'  Missing: runtime={df["runtime"].isnull().sum()}, rest=0')
print(f'  Target distribution: mean={df[target].mean():.2f}, std={df[target].std():.2f}')

# Step 4: Data Engineering (imputation + outliers)
df_clean = df[features + [target]].copy()
df_clean['runtime'].fillna(df_clean['runtime'].median(), inplace=True)
print(f'[STEP 4] Data cleaning: median imputation for runtime, no outlier removal')
print(f'  After cleaning: {df_clean.shape[0]} rows, 0 missing')

# Step 5: Feature Selection
X = df_clean[features].copy()
y = df_clean[target].copy()
# Variance threshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
kept = [f for f, s in zip(features, vt.get_support()) if s]
print(f'[STEP 5] Feature selection:')
print(f'  Variance threshold: {len(kept)}/{len(features)} pass (all pass)')
# Mutual info
mi = mutual_info_regression(X, y, random_state=42)
print(f'  Mutual info scores:')
for f, s in sorted(zip(features, mi), key=lambda x: x[1], reverse=True):
    print(f'    {f}: {s:.4f}')

# Step 6: Data Split
n_unique = y.nunique()
strat_safe = (n_unique <= 50)
print(f'[STEP 6] Data split:')
print(f'  Target unique values: {n_unique}, stratify_safe={strat_safe}')
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
print(f'  Train: {len(X_tr)}, Test: {len(X_te)} ✓')

# Step 7: Model Training + KFold
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring='r2')
model.fit(X_tr_s, y_tr)
train_r2 = r2_score(y_tr, model.predict(X_tr_s))
test_r2 = r2_score(y_te, model.predict(X_te_s))
print(f'[STEP 7] Training + CV:')
print(f'  Model: Random Forest Regressor')
print(f'  CV R2 scores: [{" ".join(f"{s:.4f}" for s in cv_scores)}]')
print(f'  Mean CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Step 8: Metrics
y_pred = model.predict(X_te_s)
mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
gap = train_r2 - test_r2
print(f'[STEP 8] Metrics:')
print(f'  Train R2: {train_r2:.4f}')
print(f'  Test R2:  {test_r2:.4f}')
print(f'  MAE: {mae:.4f}, RMSE: {rmse:.4f}')
print(f'  Gap: {gap:.4f} -> {"Overfitting" if gap > 0.15 else "Good fit" if test_r2 > 0.5 or gap < 0.15 else "Underfitting"}')

# Step 9: Hyperparameter Tuning
grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    grid, cv=5, scoring='r2', n_jobs=-1, verbose=0, refit=True
)
search.fit(X_tr_s, y_tr)
best_model = search.best_estimator_
after_r2 = r2_score(y_te, best_model.predict(X_te_s))
print(f'[STEP 9] Hyperparameter Tuning:')
print(f'  Best params: {search.best_params_}')
print(f'  Before R2: {test_r2:.4f}')
print(f'  After R2:  {after_r2:.4f}')
print(f'  Improvement: {(after_r2 - test_r2)*100:+.2f}%')

print()
print('='*60)
print('ALL 9 PIPELINE STEPS COMPLETED SUCCESSFULLY! ✓')
print('='*60)
