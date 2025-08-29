import { Injectable } from '@nestjs/common';
import { CreateMlDto } from './dto/create-ml.dto';
import { UpdateMlDto } from './dto/update-ml.dto';
import { PythonShell } from 'python-shell';
import * as path from 'path';

@Injectable()
export class MlService {
  
  async predictBankLeave(inputDataDemo) {
    // Configure options for PythonShell
    const options = {
      mode: 'text' as const,
      pythonPath: 'python3',
      scriptPath: path.join(process.cwd(), 'src', 'ml'), // Always points to src/ml
      args: [JSON.stringify(inputDataDemo)]
    };

    try{
    // Use the Promise-based API
      const results = await PythonShell.run('./trained-model/bank-leaving-prediction/bank_model_prediction.py', options);          //running the python file
      console.log("bbb")
      console.log(results)
      // console.log('Python script output:', results);
      if (!results || results.length === 0) {
        throw new Error('No output from Python script');
      }

      // Only parse the last line as JSON
      let output =  JSON.parse(results[results.length - 1]);

      if(output[0][0] >= 0.5){
        return "Customer is likely to leave the bank."
      } else {
        return "Customer is likely to stay with the bank."
      }
    

    
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  
  }




  async predictBitcoinPrice(bitcoindata){
    const options = {
      mode: 'text' as const,
      pythonPath: 'python3',
      scriptPath: path.join(process.cwd(), 'src', 'ml'),
      args: [JSON.stringify(bitcoindata)]
    };

    try{
      const results = await PythonShell.run('./trained-model/bitcoin-price-prediction/model-prediction.py', options);
      console.log("bbb")
      console.log(results)
      if (!results || results.length === 0) {
        throw new Error('No output from Python script');
      }
      // Only parse the last line as JSON if possible
      let output;
      try {
        output = JSON.parse(results[results.length - 1]);
      } catch (e) {
        // If not JSON, just return the raw result
        output = results[results.length - 1];
      }
      return output;
    } catch (error) {
      // Extract only the last line of the error message for frontend
      let errMsg = error.message;
      if (errMsg && typeof errMsg === 'string') {
        const lines = errMsg.split('\n');
        errMsg = lines[lines.length - 1] || errMsg;
      }
      console.error('Prediction error:', errMsg);
      throw new Error(`Prediction failed: ${errMsg}`);
    }
  }


  async insuranceChargesPrediction(inputData){
    // Configure options for PythonShell
    const options = {
      mode: 'text' as const,
      pythonPath: 'python3',
      scriptPath: path.join(process.cwd(), 'src', 'ml'), // Always points to src/ml
      args: [JSON.stringify(inputData)]
    };
    try {
      // Use the Promise-based API
      const results = await  PythonShell.run('./trained-model/insurance-charges-prediction/calculate-insurance.py', options)
      if (!results || results.length === 0) {
        throw new Error('No output from Python script');
      }
      // Only parse the last line as JSON if possible
      let output;
      try {
        output = JSON.parse(results[results.length - 1]);
      } catch (e) {
        // If not JSON, just return the raw result
        output = results[results.length - 1];
      }
      return output;
      
    } catch (error) {
      // Extract only the last line of the error message for frontend
      let errMsg = error.message;
      if (errMsg && typeof errMsg === 'string') {
        const lines = errMsg.split('\n');
        errMsg = lines[lines.length - 1] || errMsg;
      }
      console.error('Prediction error:', errMsg);
      throw new Error(`Prediction failed: ${errMsg}`);
    }

      
  }
}

