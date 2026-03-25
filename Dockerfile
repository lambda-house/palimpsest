FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --production
COPY server.js index.html sources.html ./
COPY src/ ./src/
USER 1000
EXPOSE 3000
CMD ["node", "server.js"]
