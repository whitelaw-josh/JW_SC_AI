import sc2
import random
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER

#Our BotAI class
class SentdeBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers() #Distribute workers evenly to mineral patches (in sc2/bot_ai.py)
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    #Method to build workers at NEXUS if affordable
    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    #Method to build pylon if supply left is less than 5 and pylon is not already being built
    #Will build pylon near nexus first and if afforadble
    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexi = self.units(NEXUS).ready
            if nexi.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexi.first)

    #Method to build assimilator within 25 units of a nexus and if affordable
    #Find worker that is available and geyser that is available to have assimilator built
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
        if self.units(NEXUS).amount < 2 and self.can_afford(NEXUS):
            await self.expand_now()

    #Method to build offensive buildings (i.e. Race = Protoss, Gateway)
    async def build_offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < 3:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

    #Method to build offensive units (i.e.  Race = Protoss, Stalkers)
    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))

    #Method to find targets for offensive units to attack 
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    #Method to make offensive units attack if more than 3 stalkers and we have seen an enemy
    #First offense, then defense
    #TODO: Conditional to check if being attacked
    #TODO: Might need to have variable (local or outside of scope of attack method) containing sum of all offensive units
    async def attack(self):
        if self.units(STALKER).amount > 15:
            for s in self.units(STALKER).idle:
                await self.do(s.attack(self.find_target(self.state)))

        elif self.units(STALKER).amount > 3:
            if len(self.known_enemy_units) > 0:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))

#Starts the game we want to run taking into the map and players we wish to add
#realtime controls speed of how game is played
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime = False)