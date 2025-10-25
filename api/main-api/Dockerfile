FROM node:22-alpine

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile --production

COPY . .
RUN yarn build

EXPOSE 8001

CMD ["yarn", "start:prod"]