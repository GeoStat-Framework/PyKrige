"""
Classification kriging
----------------------

An example of classification kriging
"""

import sys

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from pykrige.ck import ClassificationKriging

svc_model = SVC(C=0.1, gamma="auto", probability=True)
rf_model = RandomForestClassifier(n_estimators=100)
lr_model = LogisticRegression(max_iter=10000)

models = [svc_model, rf_model, lr_model]

try:
    housing = fetch_california_housing()
except PermissionError:
    # this dataset can occasionally fail to download on Windows
    sys.exit(0)

# take the first 5000 as Kriging is memory intensive
p = housing["data"][:5000, :-2]
x = housing["data"][:5000, -2:]
target = housing["target"][:5000]
discretizer = KBinsDiscretizer(encode="ordinal")
target = discretizer.fit_transform(target.reshape(-1, 1))

p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
    p, x, target, test_size=0.3, random_state=42
)

for m in models:
    print("=" * 40)
    print("classification model:", m.__class__.__name__)
    m_ck = ClassificationKriging(classification_model=m, n_closest_points=10)
    m_ck.fit(p_train, x_train, target_train)
    print(
        "Classification Score: ", m_ck.classification_model.score(p_test, target_test)
    )
    print("CK score: ", m_ck.score(p_test, x_test, target_test))
