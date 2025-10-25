import { Request } from 'express';

export interface JwtPayload {
  userId: string;
  jti: string;
}

export interface RequestWithUser extends Request {
  user: JwtPayload;
}
