import sc2
import random
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STARGATE, STALKER, VOIDRAY

#Our BotAI class
class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165 #Roughly 165 iterations (estimated)
        self.MAX_WORKERS = 50 #Max workers of 50

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers() #Distribute workers evenly to mineral patches (in sc2/bot_ai.py)
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    #Method to build workers at NEXUS if affordable
    #TODO: consider if 16 should be switched to 22 for workers in mineral patches and assimilators
    async def build_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    #Method to build pylon if supply left is less than 5 and pylon is not already being built
    #Will build pylon near nexus first and if afforadble
    #TODO: May need to update location of building pylons
    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexi = self.units(NEXUS).ready
            if nexi.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexi.first)

    #Method to build assimilator within 25 units of a nexus and if affordable
    #Find worker that is available and geyser that is available to have assimilator built
    #TODO: May need to update location of building assimilators
    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            geysers = self.state.vespene_geyser.closer_than(25.0, nexus)
            
            for gas in geysers:
                if not self.can_afford(ASSIMILATOR):
                    break

                worker = self.select_build_worker(gas.position)

                if worker is None:
                    break

                if not self.units(ASSIMILATOR).closer_than(1.0, gas).exists:
                    await self.do(worker.build(ASSIMILATOR, gas))

    #Method to build nexus if less than 2 nexi on the map and can afford it
    async def expand(self):
        if self.units(NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(NEXUS):
            await self.expand_now()

    #Method to build offensive buildings (i.e. Race = Protoss, Gateway, Stargate)
    #TODO: May need to update location of building offensive force buildings
    async def build_offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    #Method to build offensive units (i.e.  Race = Protoss, Stalkers, Voidrays)
    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gw.train(STALKER))

        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    #Method to find targets for offensive units to attack 
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    #Method to make offensive units attack and defend
    #TODO: Conditional to check if being attacked (i.e. defend)
    #TODO: Might need to have variable (local or outside of scope of attack method) containing sum of all offensive units
    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {STALKER: [15, 5],
                            VOIDRAY: [8, 3]}

        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))

#Starts the game we want to run taking into the map and players we wish to add
#realtime controls speed of how game is played
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Hard)
], realtime = False)