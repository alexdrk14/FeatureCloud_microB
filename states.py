from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role
from FeatureCloud.app.engine.app import State as op_state
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial', role=Role.BOTH)
class InitialState(AppState):

    def register(self):
        #self.register_transition('send_data', Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        #self.register_transition('aggregation', Role.COORDINATOR)
        self.register_transition('send_data', Role.BOTH)
    def run(self):
        return 'send_data'

@app_state('send_data', role=Role.BOTH)
class participant_send_data(AppState):

    def register(self):
        self.register_transition('aggregation', Role.COORDINATOR)
        #self.register_transition('get_data', Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition('performance', Role.PARTICIPANT)


    def run(self):
        """Read Data"""
        data = pd.read_csv("/mnt/input/data.csv")
        """Drop first column"""

        target = data['target']
        data.drop([data.columns.to_list()[0], 'target'], inplace=True, axis=1)
        """Split data"""
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2, shuffle=True)

        self.store("x_test", X_test)
        self.store("y_test", y_test)

        model = LogisticRegression(random_state=2)
        model.fit(X_train, y_train)
        self.store("model", model)

        msre = mean_squared_error(y_test, model.predict(X_test))
        self.log(f'Client before aggregation MSRE performance{msre}', LogLevel.DEBUG)

        self.send_data_to_coordinator({"coef": model.coef_,
                                       "intercept": model.intercept_,
                                       "classes": model.classes_})
        if self.is_coordinator:
            return 'aggregation'
        else:
            return 'performance'


@app_state('performance', role=Role.BOTH)
class participant_get_data(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        X_test = self.load("x_test")
        y_test = self.load("y_test")
        model = self.load("model")
        self.log(f"Getting data from coordinator", LogLevel.DEBUG)

        agg_data = self.await_data(n=1, unwrap=True)
        model.coef_ = agg_data["coef"]
        model.intercept_ = agg_data["intercept"]
        model.classes_ = agg_data["classes"]

        msre = mean_squared_error(y_test, model.predict(X_test))
        self.log(f'Client after aggregation MSRE performance{msre}', LogLevel.DEBUG)
        return 'terminal'  # This means we are done. If the coo


@app_state('aggregation', role=Role.COORDINATOR)
class Agg(AppState):

    def register(self):
        self.register_transition('performance', Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log("Coordinator get data from clients", LogLevel.DEBUG)
        recieved_data = self.await_data(n=len(self.clients), unwrap=True)

        self.log(f'Coordinator data from clients{recieved_data}', LogLevel.DEBUG)
        self.log("Coordinator all data recieved", LogLevel.DEBUG)
        all_classes = set()
        for item in recieved_data:
            all_classes = all_classes.union(set(item["classes"]))
        all_classes = list(all_classes)
        all_classes.sort()
        coef = np.array([0.0] * len(all_classes))
        inter = np.array([0.0] * len(all_classes))
        index = 0
        for class_v in all_classes:
            self.log(f'Single coef{recieved_data[0]["coef"]}', LogLevel.DEBUG)
            coef_new = [item["coef"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]
            inter_new = [item["intercept"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]
            self.log("New coef {} and {}".format(coef_new, sum(coef_new) / len(coef_new)), LogLevel.DEBUG)
            coef[index] = sum(coef_new) / len(coef_new)
            inter_new[index] = sum(inter_new) / len(inter_new)
            index += 1
        #coef = sum([item["coef"] for item in recieved_data]) / len(recieved_data)
        #inter = sum([item["intercept"] for item in recieved_data]) / len(recieved_data)
        self.broadcast_data({"coef": coef, "intercept": inter, "classes": np.array(all_classes)})
        return 'performance'

