# Round 4

## Algorithm challenge

In this fourth round of Prosperity a new luxury product is introduced: `MAGNIFICENT MACARONS`. `MAGNIFICENT MACARONS` are a delicacy and their value is dependent on all sorts of observable factors like hours of sun light, sugar prices (main ingredient), shipping costs, import & export tariffs and storage costs (pay 0,1 seashells per macaron). Can you find the right connections to optimize your program?

Position limits for the newly introduced products:

- `MAGNIFICENT_MACARONS`: 75
- Conversion Limit for `MAGNIFICENT_MACARONS` = 10

## Manual challenge

You’re participating in a brand new game show and have the opportunity to open up a maximum of three suitcases with great prizes in them. The whole archipelago is participating, so you’ll have to share the spoils with everyone choosing the same suitcase. Opening one suitcase is free, but for the second and third one you’ll need to pay to get inside.

Here's a breakdown of how your profit from a suitcase will be computed:
Every suitcase has its **prize multiplier** (up to 100) and number of **inhabitants** (up to 15) that will be choosing that particular suitcase. The suitcase’s total treasure is the product of the **base treasure** (10 000, same for all suitcases) and the suitcase’s specific treasure multiplier. However, the resulting amount is then divided by the sum of the inhabitants that choose the same suitcase and the percentage of opening this specific suitcase of the total number of times a suitcase has been opened (by all players).

For example, if **5 inhabitants** choose a suitcase, and **this suitcase was chosen** **10% of the total number of times a suitcase has been opened** (by all players), the prize you get from that suitcase will be divided by 15. After the division, **costs for opening a suitcase** apply (if there are any), and profit is what remains.

To help you with your decision making, here's the distribution of player's choices from **Round 2** Manual:

![Round_4_Manual_Wiki.webp](attachment:8b96766c-52b0-451c-a6e0-f012e77ad163:Round_4_Manual_Wiki.webp)

## Additional trading microstructure information:

1. ConversionObservation (detailed in “[Writing an Algorithm in Python](https://www.notion.so/17be8453a09381988c6ed45b1b597049?pvs=21)” under E-learning center) shows quotes of `MAGNIFICENT_MACARONS` offered by the chefs from Pristine Cuisine
2. To purchase 1 unit of `MAGNIFICENT_MACARONS` from Pristine Cuisine, you will purchase at askPrice, pay `TRANSPORT_FEES` and `IMPORT_TARIFF`
3. To sell 1 unit of `MAGNIFICENT_MACARONS` to Pristine Cuisine, you will sell at bidPrice, pay `TRANSPORT_FEES` and `EXPORT_TARIFF`
4. You can ONLY trade with Pristine Cuisine via the conversion request with applicable conditions as mentioned in the wiki
5. For every 1 unit of `MAGNIFICENT_MACARONS` net long position, storage cost of 0.1 Seashells per timestamp will be applied for the duration that position is held. No storage cost applicable to net short position
