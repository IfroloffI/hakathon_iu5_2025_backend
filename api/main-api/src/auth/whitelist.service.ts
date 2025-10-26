import { Injectable } from '@nestjs/common';
import { RedisService } from '../redis/redis.service';

@Injectable()
export class WhitelistService {
  private readonly TTL_SECONDS = 3600; // 1 час

  constructor(private readonly redisService: RedisService) {}

  async add(tokenId: string): Promise<void> {
    const client = await this.redisService.getClient();
    await client.set(`whitelist:${tokenId}`, '1', { EX: this.TTL_SECONDS });
  }

  async has(tokenId: string): Promise<boolean> {
    const client = await this.redisService.getClient();
    const exists = await client.exists(`whitelist:${tokenId}`);
    return exists === 1;
  }

  async del(tokenId: string): Promise<void> {
    const client = await this.redisService.getClient();
    await client.del(`whitelist:${tokenId}`);
  }
}
