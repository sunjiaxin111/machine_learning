from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
import lightgbm as lgb

gbm0 = GradientBoostingClassifier(random_state=10)
xgb1 = xgb.XGBClassifier()
lgb1 = lgb.LGBMClassifier()
lgb2 = lgb.LGBMRegressor()
rf = RandomForestClassifier()
rf2 = RandomForestRegressor()
bag = BaggingClassifier()