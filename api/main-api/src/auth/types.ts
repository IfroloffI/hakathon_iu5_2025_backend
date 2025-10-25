import { Request } from 'express';

export interface JwtPayload {
  userId: string;
  jti: string;
  username: string;
  email: string;
}

export interface RequestWithUser extends Request {
  user: JwtPayload;
}
