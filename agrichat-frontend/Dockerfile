# Use nginx to serve static files
FROM nginx:alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy your frontend files to nginx's directory
COPY . /usr/share/nginx/html

EXPOSE 80
