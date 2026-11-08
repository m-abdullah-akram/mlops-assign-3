# Task 6.1 - Docker warm-up commands (screenshot each action)

```bash
docker pull nginx:alpine
docker run -d --name web1 -p 8081:80 nginx:alpine          # run + port mapping (host 8081 -> container 80)
curl http://localhost:8081                                  # proves the mapping works
docker run -d --name web2 -p 8082:80 -v "$PWD/site:/usr/share/nginx/html:ro" nginx:alpine   # volume mount
docker logs web1                                            # read logs
docker exec -it web1 sh                                     # shell inside (type: ls /usr/share/nginx/html; exit)
docker ps ; docker ps -a                                    # list containers
docker stop web1 web2 && docker rm web1 web2                # remove containers
docker images ; docker rmi nginx:alpine                     # list and remove images
```
Create `site/index.html` first (any text) so the volume demo shows your own page.
