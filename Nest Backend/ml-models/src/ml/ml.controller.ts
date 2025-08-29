import { Controller, Get, Post, Body, Patch, Param, Delete } from '@nestjs/common';
import { MlService } from './ml.service';
import { CreateMlDto } from './dto/create-ml.dto';
import { UpdateMlDto } from './dto/update-ml.dto';

@Controller('ml')
export class MlController {
  constructor(private readonly mlService: MlService) {}


  inputDataDemo = {
    // 'CreditScore': 502,
    // 'Gender': 'Female',
    // 'Age': 42,
    // 'Tenure': 8,
    // 'Balance': 159660.8,
    // 'NumOfProducts': 3,
    // 'HasCrCard': 1,
    // 'IsActiveMember': 0,
    // 'EstimatedSalary': 113931.57,
    // 'Geography': 'France'

    'CreditScore': 699,
    'Gender': 'Female',
    'Age': 39,
    'Tenure': 1,
    'Balance': 0,
    'NumOfProducts':2,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 93826.63,
    'Geography': 'France'
  }
  
  @Get()
  getHello(): string {
    return 'Hello from ML Controller!';
  }

  bitcoindata = {
    "date": "2025-09-23"
  }
  @Get('bank-model')
  async predictBankLeave() {
    try {
      const prediction = await this.mlService.predictBankLeave(this.bitcoindata);
      
      return {
        success: true,
        prediction: prediction,
        timestamp: new Date().toISOString()

      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()

      };
    }

  }


  @Get('bitcoin-model')
  async predictBitcoinPrice() {
    try {

      const prediction = await this.mlService.predictBitcoinPrice({"date": "2025-09-23"});
      
      return {
        success: true,
        prediction: prediction,
        timestamp: new Date().toISOString()

      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()

      };
    }

  }
  
  
  @Get('insurance-model')
  async insuranceChargesPrediction() {
    try {

      const prediction = await this.mlService.insuranceChargesPrediction({
            'age': 23,
            'sex': 'male',
            'bmi': 35.9,
            'children': 2,
            'smoker': 'no',
            'region': 'southwest'
        });
      
      return {
        success: true,
        prediction: prediction,
        timestamp: new Date().toISOString()

      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()

      };
    }

  }

  // @Get('model-info')
  // getModelInfo() {
  //   const summary = this.mlService.getModelSummary();
  //   return {
  //     modelLoaded: true,
  //     summary: summary
  //   };
  // }

  @Get('health')
  healthCheck() {
    return {
      status: 'healthy',
      timestamp: new Date().toISOString()
    };
  }
}
