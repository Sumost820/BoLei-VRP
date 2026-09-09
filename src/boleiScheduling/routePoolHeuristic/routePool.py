class RoutePool:
    def __init__(self, maxSize=5000):
        self.maxSize = maxSize
        self.routes = {}
        self.routeBySignature = {}
        self.nextRouteId = 0
        self.protectedRouteIds = set()

    def addRoute(self, route, protect=False):
        if route.signature in self.routeBySignature:
            routeId = self.routeBySignature[route.signature]
            if protect:
                self.protectedRouteIds.add(routeId)
            return self.routes[routeId]

        storedRoute = route.copyWithId(self.nextRouteId)
        self.routes[storedRoute.routeId] = storedRoute
        self.routeBySignature[storedRoute.signature] = storedRoute.routeId

        if protect:
            self.protectedRouteIds.add(storedRoute.routeId)

        self.nextRouteId += 1
        self.trim()
        return storedRoute

    def addRoutes(self, routes, protect=False):
        storedRoutes = []
        for route in routes:
            storedRoutes.append(self.addRoute(route, protect=protect))
        return storedRoutes

    def protectRoutes(self, routes):
        for route in routes:
            storedRoute = self.addRoute(route, protect=True)
            self.protectedRouteIds.add(storedRoute.routeId)

    def getRoutes(self):
        return list(self.routes.values())

    def getRoute(self, routeId):
        return self.routes[routeId]

    def getSize(self):
        return len(self.routes)

    def trim(self):
        if len(self.routes) <= self.maxSize:
            return

        removable = []
        for routeId, route in self.routes.items():
            if routeId in self.protectedRouteIds:
                continue

            taskCount = max(1, route.getTaskCount())
            score = route.duration / taskCount
            removable.append((score, route.duration, routeId))

        removable.sort(reverse=True)
        removeCount = len(self.routes) - self.maxSize

        for _, _, routeId in removable[:removeCount]:
            route = self.routes.pop(routeId)
            self.routeBySignature.pop(route.signature, None)
