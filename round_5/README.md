# Round 5

## Algorithm challenge

The final round of the challenge is already here! And surprise, no new products are introduced for a change. Dull? Probably not, as you do get another treat. The island exchange now discloses to you who the counterparty is you have traded against. This means that the `counter_party` property of the `OwnTrade` object is now populated. Perhaps interesting to see if you can leverage this information to make your algorithm even more profitable?

```python
class OwnTrade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, counter_party: UserId = None) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.counter_party = counter_party
```

A mysterious bottle washed ashore with some scriptures in it. To some these might look like old rags. To others these might prove to contain essential information, or even the key to a prosperous future. Whatever you do, don't throw it back in the ocean, because it will only be a matter of time before it reappears again.
Counter party report

Amir Ant
Amir
Ant
Always busy, always building. Amir builds his trading portfolio while directing traffic in the ant colony. Always anticipating, always redirecting. No collisions on his record to this date.

Ayumi Ant
Ayumi
Ant
Precise, strategic, and fiercely competitive. Some say she suffers a major Napoleon complex, but don’t let her size fool you—Ayumi’s got big plans.

Ari Ant
Ari
Ant
Ari believes in the power of teamwork… but somehow always ends up with the best crumbs for himself. Got some great antennae to find the best deals.

Anika Ant
Anika
Ant
Is said to tickle the competition into submission when she walks all over them. Figuratively of course. Because all she does is type code, fix bugs and type more code.

Boris Beetle
Boris
Beetle
Known for his hoarding habits, Boris is always scavenging for more, bigger, and better. He doesn’t mind the occasional rhino dung he picks up. “It’s all treasure to me” is his favorite saying.

Bashir Beetle
Bashir
Beetle
A real smooth talker. The kind of trader who can sell sand in a desert. Isn’t afraid to burn his legs on scorching hot sand or a risky trading strategy.”

Bonnie Beetle
Bonnie
Beetle
A hopeless romantic with a fierce attitude. She’s pretty fed up with everyone asking what happened to Clyde. And don’t even think about asking about her trading secrets—she’ll take them to the grave.

Blue Beetle
Blue
Beetle
Legend has it that ‘Blue Monday’ was named after her. She finds it hard to be optimistic about anything—except trading and trading strategies. Continuously hums ‘I’m Blue, Da Ba Dee Da Ba Daa’ while working.

Sanjay Spider
Sanjay
Spider
Claims he actually inspired Tim Berners-Lee to build a World Wide Web. Was really disappointed by the actual web that was created. “Bits and bites won’t satisfy your hunger” is his famous saying.

Sami Spider
Sami
Spider
Prefers to work from a hammock so he can use all his legs to code. Won a speed-coding contest when he was just a teeny-tiny spider and later built a successful business with his leg dexterity training program, ‘Lexterity.’”

Sierra Spider
Sierra
Spider
Fashionista first, trader second. But don’t be fooled by her looks—she knows exactly what she’s doing, making even the ugliest trading strategies look good.

Santiago Spider
Santiago
Spider
Keeps at least six of his eight eyes glued to the monitor at all times. Anxiously awaiting the results of his latest eye exam—worried that he’s been seeing it all wrong all along.

Mikhail Mosquito
Mikhail
Mosquito
Known for his What’s All the Buzz About? podcast, Mikhail is a familiar voice in the trading community. Has a habit of only showing up when the lights go out.

Mina Mosquito
Mina
Mosquito
Has a strange obsession with blue light. Luckily, her monitor emits plenty of it, allowing her to stay glued to the screen, fine-tuning her trading strategies for days on end.

Morgan Mosquito
Morgan
Mosquito
When Morgan isn’t honing his trading skills, he’s in the lab trying to invent an alternative to blood. Not for the good of science, but out of sheer necessity—his own blood phobia causes him to pass out at the sight of even the tiniest drop.

Manuel Mosquito
Manuel
Mosquito
Well-known in the community for his colorful sombrero collection, he wears a different one every day. Claims to prefer manual trading but won’t shy away from coding when he sees an opportunity to make some extra profit.

Carlos Cockroach
Carlos
Cockroach
A seasoned trading veteran who has been around for millennia. Never cracks under pressure and takes pride in the hardships he has already endured.

Candice Cockroach
Candice
Cockroach
Hosts a wildly popular weekly trade meetup, where hundreds of cockroaches gather in dimly lit, slightly moist places to discuss trading strategies.

Carson Cockroach
Carson
Cockroach
Gets squashed in one trade, only to pop up stronger in the next. Carson plays the long game, knowing that resilience is the real currency of the market. The only thing he fears? Bright lights and sudden movements.

Cristiano Cockroach
Cristiano
Cockroach
Once destined for soccer greatness, Cristiano traded the pitch for the market. Now, instead of dribbling past defenders, he dodges bad trades. Still celebrates every win with a ‘Siuuu!’—old habits die hard.

## Manual challenge

You’ve been invited to trade on the exchange of the West Archipelago for one day only. An exclusive event and perfect opportunity to make some big final profits before the champion is crowned. Benny the Bull has granted you access to his most trusted news source: Goldberg. You’ll find all the information you need right there. Be aware that trading these foreign goods comes at a price. The more you trade in one good, the more expensive it will get. This is the final stretch. Make it count!

Trade objective
Benny the Bull has invited us to trade on the West Archipelago Exchange for one day. You have the opportunity to trade all sorts of new goods against yesterday's prices, just in time before the exchange opens for a new day. In order to get up to speed with their current market dynamics, they have granted you access to their most valuable news source: Goldberg.

Your objective is to develop a trading strategy that maximizes your profit. You have 1.000.000 SeaShells available as trading capital and can choose to either buy or sell different goods. Specify the percentage you want to buy or sell for, and make sure it doesn’t exceed your total available capital. Note that there are fees involved in these trades. The more you trade in one good, the higher the fee for that good will be.

Note that you can (re)submit new strategies as long as the round is still in progress. As soon as the round ends, the trading strategy that was submitted last will be processed.

Good luck and have fun trading!
