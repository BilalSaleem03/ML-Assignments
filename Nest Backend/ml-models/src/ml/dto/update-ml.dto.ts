import { PartialType } from '@nestjs/mapped-types';
import { CreateMlDto } from './create-ml.dto';

export class UpdateMlDto extends PartialType(CreateMlDto) {}
