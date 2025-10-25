import { Injectable } from '@nestjs/common';
import NodeCache = require('node-cache');

@Injectable()
export class WhitelistService {
  private readonly cache: NodeCache;

  constructor() {
    // TTL = 3600 сек (1 час), проверка каждые 60 сек
    this.cache = new NodeCache({ stdTTL: 3600, checkperiod: 60 });
  }

  add(tokenId: string): void {
    this.cache.set(tokenId, true);
  }

  has(tokenId: string): boolean {
    return this.cache.has(tokenId);
  }

  del(tokenId: string): void {
    this.cache.del(tokenId);
  }
}
