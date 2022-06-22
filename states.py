from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role
from FeatureCloud.app.engine.app import State as op_state

"""Splint and measurements"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""Sampling methods"""
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

"""Our implemented extra functions in order to reduce size of states.py"""
from extra import *
"""Our model class"""
from model import Model


"""Global variables for manipulating some functionality of clients"""
MAX_ITER = 50
ENTROPY_THRESHOLD = 0.08079313589591118
SAMPLING = "Over" #"Under" "Over"

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
        self.register_transition('fine-tuning', Role.BOTH)

    def run(self):
        """Read Data"""
        data, target, country = data_loading()

        """Store country"""
        self.store('country', country)
        """Store number of federated model iterations"""
        self.store("iteration", 0)

        self.log(f'Initialization of dataset for country: {country} has data shape:{data.shape}',
                 LogLevel.DEBUG)
        self.log(f'Class 0 has:{target.count(0)} and Class 1 has:{target.count(1)}',
                 LogLevel.DEBUG)

        data, drop_out, imputed = feature_NA_manager(data, ENTROPY_THRESHOLD)
        self.log(f'Imputed features:{imputed} and features to drop:{drop_out}',
                 LogLevel.DEBUG)

        """Data sampling (balancing of classes)"""
        if data.shape[0] > 10:
            if SAMPLING == "Under": # "Over"
                self.log('-------UnderSampling Method-----------',
                         LogLevel.DEBUG)
                undersampler = RandomUnderSampler(sampling_strategy='majority')
                X, Y = undersampler.fit_resample(data, target)
            elif SAMPLING == "Over":
                self.log('-------OverSampling Method-----------',
                         LogLevel.DEBUG)
                oversample = SMOTE()
                X, Y = oversample.fit_resample(data, target)
        else:
            self.log('-------Sampling is not possible-----------',
                     LogLevel.DEBUG)
            X = data
            Y = target

        self.log(f'Class 0 has:{Y.count(0)} and Class 1 has:{Y.count(1)}',
                 LogLevel.DEBUG)

        """Split data into train and test with random shuffle"""
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            stratify=Y)
        #self.log(f'----1-------',
        #        LogLevel.DEBUG)
        """Store test data into shared memory for future usage"""
        self.store("x_test", X_test)
        self.store("y_test", y_test)
        self.store("x_train", X_train)
        self.store("y_train", y_train)

        #self.log(f'----2-------',
        #         LogLevel.DEBUG)
        """Store number of samples"""
        self.store("number_of_samples", X_train.shape[0])
        #self.log(f'----3-------',
        #         LogLevel.DEBUG)
        return "fine-tuning"

@app_state('fine-tuning', role=Role.BOTH)
class model_init(AppState):

    def register(self):
        self.register_transition('TODO', Role.BOTH)

    def run(self):
        """Load train data"""
        X = self.load("x_train")
        Y = self.load("y_train")

        params = {"penalty": ['l2', 'l1'],
                  "alpha": np.arange(0.00001, 0.0005, 0.00004)}
        params_performance = {}
        for penalty in params["penalty"]:
            for alpha in params["alpha"]:
                model = Model(penalty=penalty, alpha=alpha)
                X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    stratify=Y)
                model.fit(X_train, y_train)
                mi, conf_matrix, recall = model.measure_performance(X_val, y_val)
                params_performance[f'{penalty}-{alpha}'] = mi

        """Create model , train on training data portion and store model parameters"""
        model = Model()
        model.fit(X_train, y_train)
        model_params = model.get_params()

        #imp_features = X_train.columns[model_params["coef"][0] != 0].to_list()
        #
        #self.log(f'Important:{imp_features} len: {len(imp_features)} all {X_train.shape[1]}',
        #         LogLevel.DEBUG)

        #self.log(f'----4-------',
        #         LogLevel.DEBUG)
        """
        model = LogisticRegression(max_iter=50000, random_state=2)
        model.fit(X_train, y_train)
        model_params = {"coef": model.coef_,
                        "intercept": model.intercept_,
                        "classes": model.classes_}
        """
        self.store("model", model)
        self.store("model_params", model_params)
        #self.log(f'----5-------',
        #         LogLevel.DEBUG)

        return 'p_measure'


@app_state('p_measure', role=Role.BOTH)
class p_measure(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)
        self.register_transition('send_data', Role.BOTH)

    def run(self):
        #self.log(f'----6-------',
        #         LogLevel.DEBUG)
        """Load test data from shared memory"""
        X_test = self.load("x_test")
        y_test = self.load("y_test")
        X_train = self.load("x_train")
        y_train = self.load("y_train")
        #self.log(f'----7-------',
        #         LogLevel.DEBUG)
        model = self.load("model")
        #self.log(f'----8-------',
        #         LogLevel.DEBUG)
        country = self.load("country")

        iteration = self.load("iteration")

        self.log(f"Performance of iteration:{iteration}", LogLevel.DEBUG)

        if iteration == 0:
            model_params = self.load("model_params")
            best_recall = 0.0
            best_mi = 0.0
        else:
            """Collect aggregated data from coordinator"""
            model_params = self.await_data(n=1, unwrap=True)
            best_recall = self.load("recall")
            best_mi = self.load("mi")
            old_model_params = self.load("model_params")
            model.set_params(model_params)
            """Re-train model in their own data until last iteration. 
            At the last iteration use only federated coef"""
            if iteration != (MAX_ITER - 1):
                model.partial_fit(X_train, y_train)

        #self.log(f'Test data before measuring performance{X_test}', LogLevel.DEBUG)
        #self.log(f'Classes{model.model.classes_}', LogLevel.DEBUG)

        mi, conf_matrix, recall = model.measure_performance(X_test, y_test)
        plot_confusion(conf_matrix, country, iteration)

        self.log(f'Test performance scores Recall:{recall} MI:{mi} and Confusion Matrix:{conf_matrix}', LogLevel.DEBUG)

        self.store("iteration", iteration + 1)
        if iteration > MAX_ITER:
            return 'terminal'
        if iteration != 0:
            """Check if previous model params provide better recall that new"""
            #if recall > best_recall or mi > best_mi or iteration == (MAX_ITER -1):#msre < prev_msre:
            if mi > best_mi or iteration == MAX_ITER:  # msre < prev_msre:
                """If new params provide lower error use them"""
                """In case last iteration :
                should force model to keep federated params only without re-training"""
                self.log(f'Store new params and recall',
                         LogLevel.DEBUG)
                self.store("recall", recall)
                self.store("mi", mi)
                self.store("model_params", model_params)
            else:
                self.log(f'Keep old params and recall',
                         LogLevel.DEBUG)
                model.set_params(old_model_params)
        else:
            self.store("recall", recall)
            self.store("mi", mi)
        return 'send_data'

@app_state('send_data', role=Role.BOTH)
class participant_send_data(AppState):

    def register(self):
        self.register_transition('aggregation', Role.COORDINATOR)
        self.register_transition('p_measure', Role.PARTICIPANT)

    def run(self):
        """Send data to coordinator"""
        model_params = self.load("model_params")
        """Also provide number of samples"""
        model_params["number_of_samples"] = self.load('number_of_samples')
        self.log(f'Send data to coordinator', LogLevel.DEBUG)

        self.send_data_to_coordinator(model_params)
        if self.is_coordinator:
            return 'aggregation'
        else:
            return 'p_measure'

@app_state('aggregation', role=Role.COORDINATOR)
class Agg(AppState):

    def register(self):
        self.register_transition('p_measure', Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        """Collect data from clients"""
        self.log("Coordinator get data from clients", LogLevel.DEBUG)
        #recieved_data = self.await_data(n=len(self.clients), unwrap=True)
        recieved_data = self.gather_data()

        #all_classes = set()
        #for item in recieved_data:
        #    all_classes = all_classes.union(set(item["classes"]))
        #all_classes = list(all_classes)
        #all_classes.sort()

        """Create new arrays for new coef and intercept values with only 0.0 values"""
        #coef_dim = len(recieved_data[0]["coef"][0])
        #coef = np.array([np.array([0.0] * coef_dim)] * len(all_classes) )
        #inter = np.array([0.0] * len(all_classes))
        #index = 0
        #self.log(f'Coef single{recieved_data[0]["coef"]} and classes:{recieved_data[0]["classes"]} inter:{recieved_data[0]["intercept"]}', LogLevel.DEBUG)

        #for item in recieved_data:
        #    self.log(
        #        f'Classes : {item["classes"]}',
        #        LogLevel.DEBUG)

        #all_samples = sum([client_data["number_of_samples"] for client_data in recieved_data])
        #new_coef = np.array([sum([client_data["coef"][0] * client_data["number_of_samples"] for client_data in recieved_data]) / all_samples])
        #new_inter = np.array([sum([client_data["intercept"][0] * client_data["number_of_samples"] for client_data in recieved_data]) / all_samples])


        new_coef = np.array([sum([client_data["coef"][0] for client_data in
                                  recieved_data]) / len(recieved_data)])
        new_inter = np.array([sum([client_data["intercept"][0] for client_data in
                                   recieved_data]) / len(recieved_data)])

        """
        for class_v in all_classes:
            #Compute new coef value and intercept for each class value
            coef_new = [item["coef"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]
            inter_new = [item["intercept"][np.where(item["classes"] == class_v)[0][0]] for item in recieved_data if class_v in item["classes"]]

            #Update coef and intercept values in updated structure
            coef[index] = np.array(np.array(sum(coef_new)) / len(coef_new))
            inter[index] = sum(inter_new) / len(inter_new)
            index += 1
        """
        """Propagate new coef , intercept and classes into the participants"""
        self.broadcast_data({"coef": new_coef, "intercept": new_inter, "classes": recieved_data[0]["classes"]})
        return 'p_measure'

