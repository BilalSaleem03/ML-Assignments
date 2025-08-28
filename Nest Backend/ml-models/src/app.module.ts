import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MlService } from './ml/ml.service';
import { MlModule } from './ml/ml.module';

@Module({
  imports: [MlModule],
  controllers: [AppController],
  providers: [AppService, MlService],
})
export class AppModule {}
