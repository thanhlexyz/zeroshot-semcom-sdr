class Receiver:

    def __init__(self, args, link):
        self.args = args
        self.link = link

    def receive(self):
        return self.link.pull()
