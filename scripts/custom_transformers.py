from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType

# Transformer which replaces text, identified via regex expression, with a supplied text string
class NormalizeText(Transformer, HasInputCol, HasOutputCol):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, regex_replace_string = None, normal_text = None):
        super(NormalizeText, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, regex_replace_string = None, normal_text = None):
        self.regex_replace_string = Param(self, "regex_replace_string", "")
        self.normal_text = Param(self, "normal_text", "")
        self._setDefault(regex_replace_string='', normal_text='')
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def getReplacementString(self):
        return self.getOrDefault(self.regex_replace_string)
    
    def getNormalizedText(self):
        return self.getOrDefault(self.normal_text)
    
    def _transform(self, df):
        replacement_string = self.getReplacementString()
        normalized_text = self.getNormalizedText()
        
        
        out_col = self.getOutputCol()
        in_col = self.getInputCol()
        
        df_transform = df.withColumn(out_col, f.regexp_replace(f.col(in_col), replacement_string, normalized_text))
        
        return df_transform

# Transformer which converts text to lower case
class LowerCase(Transformer, HasInputCol, HasOutputCol):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(LowerCase, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    
    def _transform(self, df):        
        
        out_col = self.getOutputCol()
        in_col = self.getInputCol()
        
        df_lower = df.withColumn(out_col, f.lower(f.col(in_col)))
        
        return df_lower
    

'''
Transformer which converts binary target variable from text to integers of 0/1
This transformer also allows for the renaming of the target variable. The
renaming defaults to "label".
'''

class BinaryTransform(Transformer, HasInputCol, HasOutputCol):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(BinaryTransform, self).__init__()
        self._setDefault(outputCol='label')
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    
    def _transform(self, df):        
        
        out_col = self.getOutputCol()
        in_col = self.getInputCol()
                
        # Replace spam/ham with 1/0
        df_binary = df.withColumn(out_col, f.when(f.col(in_col) == "spam" , 1)
                             .when(f.col(in_col) == "ham" , 0)
                             .otherwise(f.col(in_col)))
        
        #Convert 1/0 to int
        df_binary = df_binary.withColumn(out_col, f.col(out_col).cast(IntegerType()))
        
        # Drop spam/ham column
        df_binary = df_binary.drop(in_col)
        
        # Reorder dataframe so target is first column
        df_binary = df_binary.select(out_col, 'sms')

        return df_binary