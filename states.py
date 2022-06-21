from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role
from FeatureCloud.app.engine.app import State as op_state
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

MAX_ITER = 10

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial', role=Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('model_init', Role.BOTH)

    def run(self):
        return 'model_init'

@app_state('model_init', role=Role.BOTH)
class model_init(AppState):

    def register(self):
        self.register_transition('send_data', Role.BOTH)

    def run(self):
        """Read Data"""
        data = pd.read_csv("/mnt/input/data.csv")
        """Drop first column and keep target separetely"""
        target = data['target']
        data.drop([data.columns.to_list()[0], 'target'], inplace=True, axis=1)

        """Split data into train and test with random shuffle"""
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2, shuffle=True)

        """Store test data into shared memory for future usage"""
        self.store("x_test", X_test)
        self.store("y_test", y_test)

        """Create model , train on training data portion and store model parameters"""
        model = LogisticRegression(random_state=2)
        model.fit(X_train, y_train)
        model_params = {"coef": model.coef_,
                        "intercept": model.intercept_,
                        "classes": model.classes_}

        self.store("model_params", model_params)

        """measure number of federated model iterations"""
        self.store("iteration", 0)

        msre = mean_squared_error(y_test, model.predict(X_test))
        self.store("model_error", msre)
        self.log(f'Client before aggregation MSRE performance{msre}', LogLevel.DEBUG)

        return 'send_data'


@app_state('send_data', role=Role.BOTH)
class participant_send_data(AppState):

    def register(self):
        self.register_transition('aggregation', Role.COORDINATOR)
        self.register_transition('performance', Role.PARTICIPANT)

    def run(self):
        """Send data to coordinator"""
        model_params = self.load("model_params")
        self.log(f'Send data to coordinator', LogLevel.DEBUG)

        self.send_data_to_coordinator(model_params)
        if self.is_coordinator:
            return 'aggregation'
        else:
            return 'performance'


@app_state('performance', role=Role.BOTH)
class participant_get_data(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition('send_data', Role.BOTH)

    def run(self):
        """Load test data from shared memory"""
        X_test = self.load("x_test")
        y_test = self.load("y_test")
        iteration = self.load("iteration")

        self.log(f"Getting data from coordinator", LogLevel.DEBUG)

        """Collect aggregated data from coordinator"""
        model_params = self.await_data(n=1, unwrap=True)

        """Kepp coef and intercept tables only for classes that model know"""
        prev_model_params = self.load("model_params")
        classes = prev_model_params["classes"]

        ind = [np.where(model_params["classes"] == value)[0][0] for value in classes]
        model_params["coef"] = model_params["coef"][ind]
        model_params["intercept"] = model_params["intercept"][ind]
        model_params["classes"] = model_params["classes"][ind]

        """Update model coef, intercept and class"""
        model = LogisticRegression(random_state=2)
        model.coef_ = model_params["coef"]
        model.intercept_ = model_params["intercept"]
        model.classes_ = model_params["classes"]

        msre = mean_squared_error(y_test, model.predict(X_test))
        prev_msre = self.load("model_error")
        self.log(f'Client after aggregation MSRE before:{prev_msre} and after:{msre} for iteration{iteration}', LogLevel.DEBUG)


        msre = mean_squared_error(y_test, model.predict(X_test))


        if iteration < MAX_ITER:
            self.store("iteration", iteration + 1)
            """Check if previous model params provide lower error"""
            if msre < prev_msre:
                """If new params provide lower error use them"""
                self.log(f'!!Store new model error and model parameters',
                         LogLevel.DEBUG)
                self.store("model_error", msre)
                self.store("model_params", model_params)
            else:
                self.log(f'!!Keep old model error and model parameters',
                         LogLevel.DEBUG)
            return 'send_data'
        else:
            return 'terminal'

        if prev_msre < msre and iteration < 10:
            """Model perform better with previous weights"""
            model_params = self.load("model_params")
            self.store("iteration", iteration + 1)
        else:
            return 'terminal'  # This means we are done. If the coo


@app_state('aggregation', role=Role.COORDINATOR)
class Agg(AppState):

    def register(self):
        self.register_transition('performance', Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        """Collect data from clients"""
        self.log("Coordinator get data from clients", LogLevel.DEBUG)
        #recieved_data = self.await_data(n=len(self.clients), unwrap=True)
        recieved_data = self.gather_data()

        all_classes = set()
        for item in recieved_data:
            all_classes = all_classes.union(set(item["classes"]))
        all_classes = list(all_classes)
        all_classes.sort()

        """Create new arrays for new coef and intercept values with only 0.0 values"""
        coef_dim = len(recieved_data[0]["coef"][0])
        coef = np.array([np.array([0.0] * coef_dim)] * len(all_classes) )
        inter = np.array([0.0] * len(all_classes))
        index = 0
        for class_v in all_classes:
            """Compute new coef value and intercept for each class value"""
            coef_new = [item["coef"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]
            inter_new = [item["intercept"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]

            """Update coef and intercept values in updated structure"""
            coef[index] = np.array(np.array(sum(coef_new)) / len(coef_new))
            inter[index] = sum(inter_new) / len(inter_new)
            index += 1
        """Propagate new coef , intercept and classes into the participants"""
        self.broadcast_data({"coef": coef, "intercept": inter, "classes": np.array(all_classes)})
        return 'performance'

