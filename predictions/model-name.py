class rms_titanic(BaseClass):

    def pre_process(self, input):
        # Transform json to dataframe and pre-process it
        home_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data = input
        data_extra = joblib.load(os.path.join(os.path.join(os.path.join(home_path, "data_extra"), "rms_titanic"),"rms_titanic_col_extra.pkl"))
        df = pd.get_dummies(pd.DataFrame(data))
        output1 = df.reindex(columns=data_extra, fill_value=0)
        print(response)
        return output1

    def prediction(self, output1, model):
        output2 = model.predict(output1)
        return output2

    def post_process(self, output2):
        output_list=output2.tolist()
        output_final=[]
        for a in output_list:
            element= {"Result": a}
            output_final.append(element)
        return output_final




"""
o	In this script, you have to declare an object class that contain three methods: always declare a class called as the model (“model_name”) that inherits the “BaseClass” as in the example. This class must contain the 3 mandatory methods with the corresponding argument for input as specified below: 
	def pre_process(self, json_input): get the input data from the request in json format. In this function, you should do all pre-processing for the input data. For example, you can add more data to feed the input data. Return a dataframe. 
	def prediction(self, df, model): Get the dataframe from the pre_process method and the model trained (pkl file loaded in memory). Here is where you must implement de prediction called for the trained model (model.pkl charged in memory). Return and array, dataframe or whatever return the prediction of the model. 
	def post_process(self, output): Get the output from the prediction method and parse it to the output format configured in the model configuration. 

"""