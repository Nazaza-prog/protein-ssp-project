let games = [
    {id: 1, title: "The Witcher 3", genre: "RPG", price: 800, rating: 10, available: true},
    {id: 2, title: "Cyberpunk 2077", genre: "RPG", price: 1000, rating: 8, available: true},
    {id: 3, title: "Counter-Strike 2", genre: "Shooter", price: 0, rating: 7, available: true},
    {id: 4, title: "Dota 2", genre: "MOBA", price: 0, rating: 8, available: false},
    {id: 5, title: "Minecraft", genre: "Sandbox", price: 600, rating: 9, available: true},
    {id: 6, title: "Elden Ring", genre: "Souls-like", price: 1200, rating: 10, available: true}
];

console.log("Список усіх ігор");
for (let game of games) {
    console.log(`${game.title} | Жанр: ${game.genre} | Ціна: ${game.price} грн | Рейтинг: ${game.rating}`);
}

console.log(`\nЗагальна кількість ігор у каталозі: ${games.length}`);

let newTitle = prompt("Введіть назву нової гри:");
let newGenre = prompt("Введіть жанр нової гри:");
let newPrice = Number(prompt("Введіть ціну нової гри:"));
let newRating = Number(prompt("Введіть рейтинг гри (від 1 до 10):"));

let newGame = {
    id: games.length + 1,
    title: newTitle,
    genre: newGenre,
    price: newPrice,
    rating: newRating,
    available: true
};
games.push(newGame);
console.log("\nКаталог після додавання нової гри");
console.log(games);

let confirmDelete = confirm("Ви дійсно хочете видалити останню гру з каталогу?");
if (confirmDelete) {
    let removedGame = games.pop();
    alert(`Гру "${removedGame.title}" успішно видалено`);
} else {
    alert("Видалення скасовано");
}

let searchTitle = prompt("Введіть назву гри для пошуку:");
let foundGame = games.find(game => game.title.toLowerCase() === searchTitle.toLowerCase());

if (foundGame !== undefined) {
    alert(`Знайдено гру!\nНазва: ${foundGame.title}\nЦіна: ${foundGame.price} грн\nРейтинг: ${foundGame.rating}`);
} else {
    alert("Такої гри немає в каталозі");
}

let maxPrice = Number(prompt("Введіть максимальну ціну для фільтрації ігор:"));
let budgetGames = games.filter(game => game.price <= maxPrice);
console.log(`\nІгри дешевші за ${maxPrice} грн`);
console.log(budgetGames);

let topRatedGames = games.filter(game => game.rating >= 8);
console.log("\nІгри з високим рейтингом (від 8 балів)");
console.log(topRatedGames);

let gameTitles = games.map(game => game.title);
console.log("\nНазви всіх ігор");
console.log(gameTitles);

let gamesDiscount = games.map(game => {
    return {
        id: game.id,
        title: game.title,
        genre: game.genre,
        price: Math.round(game.price * 0.85),
        rating: game.rating,
        available: game.available
    };
});
console.log("\nКаталог зі знижкою 15%");
console.log(gamesDiscount);

let sortedByPrice = games.slice().sort((a, b) => a.price - b.price);
console.log("\nІгри, відсортовані за ціною (зростання)");
console.log(sortedByPrice);

const myName = "Олександр";
let nameLength = myName.length;
let myBudget = nameLength * 100;

console.log(`\nІндивідуальне завдання`);
console.log(`Ім'я: ${myName} (символів: ${nameLength})`);
console.log(`Ваш власний бюджет: ${myBudget} грн`);

let availableGames = games.filter(game => game.available === true);
let totalCost = availableGames.reduce((sum, game) => sum + game.price, 0);

console.log("Доступні для купівлі ігри:");
availableGames.forEach(game => {
    console.log(`- ${game.title}: ${game.price} грн`);
});
console.log(`Загальна вартість усіх доступних ігор: ${totalCost} грн`);

if (totalCost > myBudget) {
    alert(`Бюджету недостатньо! Ваші кошти: ${myBudget} грн, а вартість ігор: ${totalCost} грн.`);
} else {
    let remainder = myBudget - totalCost;
    alert(`Ви можете придбати всі доступні ігри! Залишок коштів: ${remainder} грн.`);
}