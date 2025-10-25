import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: process.env.CORS_ORIGIN,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'Accept', 'X-Requested-With'],
    credentials: true,
    maxAge: 86400,
  });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true, // удаляет свойства, не в DTO
      forbidNonWhitelisted: true, // возвращает ошибку, если есть лишние поля
      transform: true, // преобразует типы (например, string в number)
      stopAtFirstError: true, // останавливает на первой ошибке
    })
  );

  await app.listen(process.env.PORT ?? 8001);
}
bootstrap();
